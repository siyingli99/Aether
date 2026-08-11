"""Self-contained refinement script for AETHER.

Runs inside the ``difix3d`` conda env. Reads images from ``--input_dir``,
runs Difix (NVIDIA/difix or NVIDIA/difix_ref) one-step diffusion refinement,
and writes refined images (same basenames) into ``--output_dir``. Optionally
writes ``output.mp4`` from the refined frames.

The image is resized to dimensions that are multiples of 8 while preserving
aspect ratio (Difix uses an SD-VAE which requires 8x). The result is then
resized back to the original resolution for downstream use.

Usage (called by ``aether/refine/difix.py``):
    python scripts/refine_dir.py \
        --input_dir  PATH/vad \
        --output_dir PATH/vad_refined \
        --max_side   1024 \
        [--ref_dir PATH/vad]   # optional MV-UNet conditioning
        [--video] [--fps 10]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob

import numpy as np
import torch
from PIL import Image

# Make sibling `src/` importable so we get the repo's pipeline_difix.py
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from pipeline_difix import DifixPipeline  # noqa: E402


_VALID_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _round8(n: int) -> int:
    return max(8, int(round(n / 8.0)) * 8)


def _fit_dims(w: int, h: int, max_side: int) -> tuple[int, int]:
    s = max_side / float(max(w, h))
    return _round8(int(w * s)), _round8(int(h * s))


def _list_images(d: str) -> list[str]:
    out: list[str] = []
    for ext in _VALID_EXT:
        out.extend(glob(os.path.join(d, f"*{ext}")))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ref_dir", default=None,
                    help="Optional MV-UNet reference frame dir.")
    ap.add_argument("--model_name", default="nvidia/difix",
                    help="Switch to nvidia/difix_ref when --ref_dir is set.")
    ap.add_argument("--prompt", default="remove degradation")
    ap.add_argument("--timestep", type=int, default=199)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_side", type=int, default=1024,
                    help="Long-edge resolution Difix runs at (multiple of 8).")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only refine the first N frames (debug).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = _list_images(args.input_dir)
    if args.limit > 0:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit(f"[refine_dir] no images in {args.input_dir}")
    print(f"[refine_dir] {len(inputs)} frames from {args.input_dir}")

    refs: list[str] | None = None
    if args.ref_dir:
        refs = _list_images(args.ref_dir)
        if len(refs) != len(inputs):
            raise SystemExit(
                f"[refine_dir] ref count {len(refs)} != input count {len(inputs)}"
            )
        if args.model_name == "nvidia/difix":
            args.model_name = "nvidia/difix_ref"
        print(f"[refine_dir] using ref dir {args.ref_dir} model={args.model_name}")

    print(f"[refine_dir] loading {args.model_name} ...")
    pipe = DifixPipeline.from_pretrained(args.model_name, trust_remote_code=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device="cuda").manual_seed(int(args.seed))

    # Warm-up call to compile triton / VAE kernels so per-frame timing is fair.
    # difix_ref is a MV-UNet (num_views=2): it needs a reference image too,
    # otherwise the (b v) rearrange fails on a single-image batch.
    dummy = Image.new("RGB", (256, 256), color=0)
    warm_kwargs = dict(num_inference_steps=1, timesteps=[args.timestep],
                       guidance_scale=0.0, generator=gen)
    if refs is not None:
        warm_kwargs["ref_image"] = dummy
    _ = pipe(args.prompt, image=dummy, **warm_kwargs).images[0]
    torch.cuda.synchronize()

    timings: list[float] = []
    for i, p in enumerate(inputs):
        img = Image.open(p).convert("RGB")
        ow, oh = img.size
        tw, th = _fit_dims(ow, oh, args.max_side)
        img_r = img.resize((tw, th), Image.BICUBIC)

        ref = None
        if refs is not None:
            ref = Image.open(refs[i]).convert("RGB").resize(
                (tw, th), Image.BICUBIC
            )

        torch.cuda.synchronize(); t0 = time.perf_counter()
        kwargs = dict(
            prompt=args.prompt,
            image=img_r,
            num_inference_steps=1,
            timesteps=[args.timestep],
            guidance_scale=0.0,
            generator=gen,
        )
        if ref is not None:
            kwargs["ref_image"] = ref
        out = pipe(**kwargs).images[0]
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        timings.append(dt)

        # Restore native resolution so downstream consumers keep their pixel grid.
        if (out.size[0], out.size[1]) != (ow, oh):
            out = out.resize((ow, oh), Image.BICUBIC)

        out_path = os.path.join(args.output_dir, os.path.basename(p))
        # Always save as PNG-quality; preserve original ext.
        out.save(out_path, quality=95)
        print(f"[refine_dir] [{i+1}/{len(inputs)}] {dt*1000:6.1f} ms  -> {out_path}")

    arr = np.array(timings)
    print(f"[refine_dir] per-frame ms  mean={arr.mean()*1000:.1f}  "
          f"p50={np.percentile(arr,50)*1000:.1f}  "
          f"p90={np.percentile(arr,90)*1000:.1f}")

    if args.video:
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio
        vid = os.path.join(args.output_dir, "video.mp4")
        writer = imageio.get_writer(vid, fps=args.fps, codec="libx264",
                                    quality=8, macro_block_size=1)
        for p in _list_images(args.output_dir):
            writer.append_data(np.asarray(Image.open(p).convert("RGB")))
        writer.close()
        print(f"[refine_dir] wrote {vid}")

    # Drop a small json sidecar so AETHER can read perf.
    import json
    sidecar = os.path.join(args.output_dir, "refine_perf.json")
    with open(sidecar, "w") as f:
        json.dump({
            "model": args.model_name,
            "frames": len(inputs),
            "max_side": args.max_side,
            "timestep": args.timestep,
            "ms_per_frame_mean": float(arr.mean() * 1000),
            "ms_per_frame_p50":  float(np.percentile(arr, 50) * 1000),
            "ms_per_frame_p90":  float(np.percentile(arr, 90) * 1000),
        }, f, indent=2)
    print(f"[refine_dir] wrote {sidecar}")


if __name__ == "__main__":
    main()
