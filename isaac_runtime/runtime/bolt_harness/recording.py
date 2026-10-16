"""Continuous, simulation-time RGB episode recording."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


def _open_video_writer(path: Path, fps: float) -> Any:
    """Use the same installed imageio/FFmpeg stack as the repository recorder."""
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )


class EpisodeRGBVideoRecorder:
    """Write real RGB frames at one fixed simulation-time cadence for one episode.

    The caller is responsible for sampling only newly rendered camera frames and
    closing the recorder in its episode-level ``finally`` block.
    """

    def __init__(self, video_path: str | Path, *, fps: float) -> None:
        if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(float(fps)) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        self.video_path = Path(video_path)
        self.timestamps_path = self.video_path.with_suffix(".timestamps.json")
        self.fps = float(fps)
        self.period_s = 1.0 / self.fps
        self._cadence_tolerance_s = max(1e-6, self.period_s * 1e-3)
        self._writer: Any | None = None
        self._shape: tuple[int, int, int] | None = None
        self._sim_times_s: list[float] = []
        self._closed = False
        self.error: str | None = None

    @property
    def frame_count(self) -> int:
        return len(self._sim_times_s)

    def _reject(self, error_type: type[Exception], message: str) -> None:
        self.error = self.error or f"{error_type.__name__}: {message}"
        raise error_type(message)

    def append(self, rgb: np.ndarray, sim_time_s: float) -> None:
        """Append one camera RGB array with its acquisition simulation time."""
        if self._closed:
            raise RuntimeError("episode video recorder is closed")
        if not isinstance(rgb, np.ndarray):
            self._reject(TypeError, "rgb must be a NumPy array from the camera")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            self._reject(ValueError, "rgb must have shape (height, width, 3) and dtype uint8")
        if rgb.shape[0] < 1 or rgb.shape[1] < 1:
            self._reject(ValueError, "rgb dimensions must be non-empty")
        if isinstance(sim_time_s, bool) or not isinstance(sim_time_s, Real):
            self._reject(ValueError, "sim_time_s must be a finite non-negative number")
        sim_time_s = float(sim_time_s)
        if not math.isfinite(sim_time_s) or sim_time_s < 0:
            self._reject(ValueError, "sim_time_s must be a finite non-negative number")

        shape = tuple(int(value) for value in rgb.shape)
        if self._shape is not None and shape != self._shape:
            self._reject(ValueError, f"camera frame shape changed from {self._shape} to {shape}")
        if self._sim_times_s:
            delta = sim_time_s - self._sim_times_s[-1]
            if delta <= 0:
                self._reject(ValueError, "camera simulation timestamps must be strictly increasing")
            if abs(delta - self.period_s) > self._cadence_tolerance_s:
                self._reject(
                    ValueError,
                    f"camera frame cadence {delta:.9f}s does not match configured period "
                    f"{self.period_s:.9f}s",
                )

        if self._writer is None:
            try:
                self.video_path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = _open_video_writer(self.video_path, self.fps)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise
        try:
            self._writer.append_data(np.ascontiguousarray(rgb))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        self._shape = shape
        self._sim_times_s.append(sim_time_s)

    def close(self) -> str | None:
        """Finalize the video and write its per-frame simulation timestamps."""
        if self._closed:
            return self.error
        self._closed = True
        if self.frame_count == 0:
            self.error = self.error or "no camera RGB frames were recorded"
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                self.error = self.error or f"{type(exc).__name__}: {exc}"
            finally:
                self._writer = None

        manifest = {
            "schema": "episode-rgb-video-timestamps-v1",
            "video_file": self.video_path.name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "rgb_shape_hwc": list(self._shape) if self._shape is not None else None,
            "sim_time_s": self._sim_times_s,
            "video_finalized": self.error is None,
            "error": self.error,
        }
        try:
            self.timestamps_path.parent.mkdir(parents=True, exist_ok=True)
            self.timestamps_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        return self.error

    def __enter__(self) -> EpisodeRGBVideoRecorder:
        if self._closed:
            raise RuntimeError("episode video recorder is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        error = self.close()
        if error is not None and exc_type is None:
            raise RuntimeError(f"episode video recording failed: {error}")
        return False


__all__ = ["EpisodeRGBVideoRecorder"]
