"""Streaming MP4 writer and frame composition helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SURROUND = (["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"])


class VideoWriter:
    """Append frames to an MP4 (H.264, yuv420p) without keeping them in RAM."""

    def __init__(self, path: str | Path, fps: float) -> None:
        self.path, self.fps = str(path), float(fps)
        self.frames = 0
        self._writer = None
        self._size = None

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        size = (int(frame.shape[1]) // 2 * 2, int(frame.shape[0]) // 2 * 2)
        if self._writer is None:
            from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
            self._size = size
            self._writer = FFMPEG_VideoWriter(self.path, size, self.fps, codec="libx264",
                                              ffmpeg_params=["-pix_fmt", "yuv420p"])
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = np.asarray(Image.fromarray(frame).resize(self._size, Image.Resampling.LANCZOS))
        self._writer.write_frame(frame)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def camera_grid(images: dict[str, np.ndarray]) -> np.ndarray:
    """Surround layout for the six nuScenes cameras, else a horizontal strip."""
    if all(c in images for row in SURROUND for c in row):
        return np.concatenate([np.concatenate([images[c] for c in row], axis=1)
                               for row in SURROUND], axis=0)
    first = next(iter(images.values()))
    h = first.shape[0]
    return np.concatenate([resize_height(images[c], h) for c in images], axis=1)


def resize_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(2, int(round(frame.shape[1] * height / frame.shape[0])))
    return np.asarray(Image.fromarray(frame).resize((width, int(height)), Image.Resampling.LANCZOS))


def font(size: int = 14):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(frame: np.ndarray, text: str, size: int = 16) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    f = font(size)
    box = draw.textbbox((0, 0), text, font=f)
    draw.rounded_rectangle((10, 10, 26 + box[2], 22 + box[3]), radius=6, fill=(15, 18, 24))
    draw.text((18, 15), text, font=f, fill=(235, 238, 242))
    return np.asarray(image)
