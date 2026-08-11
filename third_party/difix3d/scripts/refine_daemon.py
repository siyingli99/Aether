"""Long-lived Difix3D refinement daemon (runs inside the ``difix3d`` env).

Binary IPC protocol (v2, batched + bf16)
----------------------------------------
Per request, the parent process sends ONE JSON header line on stdin
(terminated by ``\n``), optionally followed by binary payload bytes:

    {"cmd": "refine", "n": <N>, "H": <H>, "W": <W>,
     "max_side": <int>, "timestep": <int>, "dtype": "bf16"|"fp16"|"fp32"}
    <N * H * W * 3 bytes uint8 RGB, row-major, contiguous>

Daemon replies with one JSON header line on stdout, followed by the
refined images as N*H*W*3 uint8 bytes:

    {"ok": true, "n": <N>, "H": <H>, "W": <W>, "ms": <float>}
    <N * H * W * 3 bytes uint8 RGB>

All N cameras share a single batched ``pipe(...)`` call (one UNet
forward, one VAE encode/decode), which is far cheaper than N serial
calls. ``max_side`` is the longest-edge resize target before VAE encode.

Other commands (no payload):
    {"cmd": "exit"}            -> shutdown
    {"cmd": "ping"}            -> {"ok": true, "pong": <float>}

Initial event (auto-emitted once the model is hot):
    {"ok": true, "event": "ready"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from pipeline_difix import DifixPipeline  # noqa: E402


_STDIN_B  = sys.stdin.buffer
_STDOUT_B = sys.stdout.buffer


def _round8(n: int) -> int:
    return max(8, int(round(n / 8.0)) * 8)


def _fit_dims(w: int, h: int, max_side: int) -> tuple[int, int]:
    s = max_side / float(max(w, h))
    return _round8(int(w * s)), _round8(int(h * s))


def _emit_header(obj: dict) -> None:
    _STDOUT_B.write((json.dumps(obj) + "\n").encode("utf-8"))
    _STDOUT_B.flush()


def _emit_with_payload(obj: dict, payload: bytes) -> None:
    line = (json.dumps(obj) + "\n").encode("utf-8")
    _STDOUT_B.write(line)
    _STDOUT_B.write(payload)
    _STDOUT_B.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[refine_daemon] {msg}\n")
    sys.stderr.flush()


def _read_exact(n: int) -> bytes:
    """Read exactly n bytes from stdin, retrying on partial reads."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = _STDIN_B.read(remaining)
        if not chunk:
            raise EOFError(f"stdin closed with {remaining}/{n} bytes pending")
    
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_header() -> dict | None:
    line = _STDIN_B.readline()
    if not line:
        return None
    s = line.decode("utf-8", errors="replace").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception as e:
        _emit_header({"ok": False, "err": f"bad json: {e!s}"})
        return {}


@torch.no_grad()
def _refine_batch(pipe, gen, prompt: str, dtype: torch.dtype,
                  imgs: np.ndarray, max_side: int, timestep: int
                  ) -> tuple[np.ndarray, float]:
    """imgs: (N, H, W, 3) uint8.  Returns (N, H, W, 3) uint8, elapsed ms."""
    n, oh, ow, _ = imgs.shape
    tw, th = _fit_dims(ow, oh, max_side)

    # Convert to PIL list (DifixPipeline preprocess expects PIL or tensor).
    if (tw, th) != (ow, oh):
        pil_list = [
            Image.fromarray(imgs[i]).resize((tw, th), Image.BICUBIC)
            for i in range(n)
        ]
    else:
        pil_list = [Image.fromarray(imgs[i]) for i in range(n)]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(
        prompt=[prompt] * n,
        image=pil_list,
        num_inference_steps=1,
        timesteps=[timestep],
        guidance_scale=0.0,
        generator=gen,
        height=th, width=tw,
        output_type="np",          # returns np.float32 in [0,1], shape (N,H,W,3)
    ).images
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out = np.clip(out, 0.0, 1.0)
    out_u8 = (out * 255.0 + 0.5).astype(np.uint8)

    if (tw, th) != (ow, oh):
        # Resize back per-image with PIL (bicubic).
        resized = np.empty((n, oh, ow, 3), dtype=np.uint8)
        for i in range(n):
            resized[i] = np.asarray(
                Image.fromarray(out_u8[i]).resize((ow, oh), Image.BICUBIC)
            )
        out_u8 = resized
    return out_u8, elapsed_ms


def _handle(pipe, gen, prompt: str, dtype: torch.dtype, req: dict) -> None:
    cmd = req.get("cmd", "refine")
    if cmd == "ping":
        _emit_header({"ok": True, "pong": time.time()})
        return
    if cmd != "refine":
        _emit_header({"ok": False, "err": f"unknown cmd: {cmd!r}"})
        return

    n  = int(req["n"])
    H  = int(req["H"])
    W  = int(req["W"])
    max_side = int(req.get("max_side", 512))
    timestep = int(req.get("timestep", 199))

    nbytes = n * H * W * 3
    try:
        payload = _read_exact(nbytes)
    except EOFError as e:
        _emit_header({"ok": False, "err": str(e)})
        return

    imgs = np.frombuffer(payload, dtype=np.uint8).reshape(n, H, W, 3).copy()
    out_u8, ms = _refine_batch(pipe, gen, prompt, dtype,
                               imgs, max_side, timestep)
    _emit_with_payload(
        {"ok": True, "n": n, "H": H, "W": W, "ms": float(ms)},
        out_u8.tobytes(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="nvidia/difix")
    ap.add_argument("--prompt", default="remove degradation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    _log(f"loading {args.model_name} (dtype={args.dtype}) ...")
    torch_dtype = {"bf16": torch.bfloat16,
                   "fp16": torch.float16,
                   "fp32": torch.float32}[args.dtype]

    pipe = DifixPipeline.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
    )
    pipe.to("cuda")
    # Some pipelines keep VAE/text_encoder in fp32 by default; force cast.
    try:
        pipe.unet.to(torch_dtype)
        pipe.vae.to(torch_dtype)
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.to(torch_dtype)
    except Exception as e:
        _log(f"warning: dtype cast partial: {e!r}")

    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device="cuda").manual_seed(int(args.seed))

    # Warmup at a representative size (1 image, 512 max_side).
    dummy = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    try:
        _refine_batch(pipe, gen, args.prompt, torch_dtype,
                      dummy, max_side=256, timestep=199)
    except Exception as e:
        _log(f"warmup failed: {e!r}")
    torch.cuda.synchronize()
    _log("ready")
    _emit_header({"ok": True, "event": "ready"})

    while True:
        req = _read_header()
        if req is None:
            _log("stdin closed; exit")
            return
        if req == {}:
            continue  # bad json already replied
        if req.get("cmd") == "exit":
            _log("exit")
            return
        try:
            _handle(pipe, gen, args.prompt, torch_dtype, req)
        except Exception as e:
            import traceback
            _emit_header({"ok": False,
                          "err": f"{e!s}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
