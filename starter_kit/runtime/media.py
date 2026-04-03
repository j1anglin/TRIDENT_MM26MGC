from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

try:
    from decord import VideoReader, cpu as decord_cpu  # type: ignore

    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import imageio.v3 as iio  # type: ignore

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    import torchvision.transforms as T  # noqa: F401

    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def require_video_support() -> None:
    missing = []
    if not HAS_TORCHVISION:
        missing.append("torchvision")
    if not (HAS_DECORD or HAS_IMAGEIO):
        missing.append("decord or imageio[v3]")
    if missing:
        hint = " and ".join(missing)
        raise ImportError(
            f"Video evaluation requires {hint}. Install the missing dependencies before running video-capable tasks."
        )


def _sample_indices(total: int, max_frames: int) -> List[int]:
    if total <= 0:
        return []
    if total <= max_frames:
        return list(range(total))
    step = max(1, math.floor(total / max_frames))
    return list(range(0, total, step))[:max_frames]


def load_video_frames_raw(
    video_path: Path,
    num_segments: int = 8,
    max_frames: Optional[int] = None,
):
    from PIL import Image as PILImage

    frames: List[PILImage.Image] = []

    if HAS_DECORD:
        vr = VideoReader(str(video_path), ctx=decord_cpu(0), num_threads=1)
        total = len(vr)
        idxs = _sample_indices(total, max_frames or num_segments)
        for i in idxs:
            frames.append(PILImage.fromarray(vr[i].asnumpy()).convert("RGB"))
    elif HAS_IMAGEIO:
        meta = {}
        try:
            meta = iio.immeta(str(video_path))
        except Exception:
            pass
        nframes = meta.get("nframes", None)
        if isinstance(nframes, int) and nframes > 0:
            idxs = _sample_indices(nframes, max_frames or num_segments)
        else:
            idxs = list(range(num_segments))
        for i in idxs:
            try:
                frames.append(PILImage.fromarray(iio.imread(str(video_path), index=i)).convert("RGB"))
            except Exception:
                break
    else:
        raise ImportError("Need decord or imageio for video decoding. Install one of them to enable video support.")

    if not frames:
        raise ValueError(f"Failed to decode frames from {video_path}")
    return frames
