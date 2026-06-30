"""Media helpers for probing videos, extracting clips, frames, and audio.

Ported from video_tools/media.py — same ffmpeg/ffprobe logic.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ZoomBox = tuple[float, float, float, float]


class MediaToolError(RuntimeError):
    """Raised when ffmpeg/ffprobe operations fail."""


def _require_binary(name: str) -> None:
    if shutil.which(name):
        return
    raise MediaToolError(f"Required binary '{name}' was not found in PATH")


def ffprobe_metadata(video_path: str) -> dict:
    """Return ffprobe metadata as JSON dict."""
    _require_binary("ffprobe")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,filename,format_name,size:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffprobe failed")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError("ffprobe returned invalid JSON") from exc


def video_duration_seconds(video_path: str) -> float:
    """Return duration in seconds."""
    meta = ffprobe_metadata(video_path)
    duration = meta.get("format", {}).get("duration")
    try:
        return float(duration)
    except (TypeError, ValueError):
        raise MediaToolError(f"Could not parse duration: {duration!r}")


def get_video_info(video_path: str) -> dict:
    """Get video metadata: duration, resolution, codec, file size."""
    meta = ffprobe_metadata(video_path)
    fmt = meta.get("format", {})
    streams = meta.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    frame_rate = video_stream.get("r_frame_rate")
    fps: float | None = None
    if isinstance(frame_rate, str) and frame_rate:
        try:
            num_str, denom_str = frame_rate.split("/", 1)
            num = float(num_str)
            denom = float(denom_str)
            if denom:
                fps = round(num / denom, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = None

    return {
        "duration_s": float(fmt.get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name") or video_stream.get("codec_type", "unknown"),
        "fps": fps,
        "container": fmt.get("format_name", "unknown"),
        "file_size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 2),
    }


def extract_clip(video_path: str, start_s: float, end_s: float, output_path: str) -> str:
    """Extract a clip using ffmpeg and return output path."""
    _require_binary("ffmpeg")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.1, float(end_s))

    src = Path(video_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg clip extraction failed")
    return str(dst)


def extract_frame(video_path: str, at_s: float, output_path: str) -> str:
    """Extract a single frame at an absolute timestamp."""
    _require_binary("ffmpeg")
    at_s = max(0.0, float(at_s))

    src = Path(video_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{at_s:.3f}",
        "-i", str(src),
        "-frames:v", "1",
        "-q:v", "2",
        str(dst),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg frame extraction failed")
    return str(dst)


def validate_zoom_box(box: object) -> ZoomBox:
    """Validate a 0-1000 coordinate box and return floats."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("zoom_box must be a 4-item sequence: (x1, y1, x2, y2)")

    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        raise ValueError("zoom_box coordinates must be numbers") from None

    if min(x1, y1, x2, y2) < 0 or max(x1, y1, x2, y2) > 1000:
        raise ValueError("zoom_box coordinates must be between 0 and 1000")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("zoom_box must have x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def compose_zoom_box(parent: ZoomBox | None, child: object | None) -> ZoomBox | None:
    """Compose a child 0-1000 box inside an optional parent box."""
    if child is None:
        return parent
    child_box = validate_zoom_box(child)
    if parent is None:
        return child_box

    px1, py1, px2, py2 = parent
    cx1, cy1, cx2, cy2 = child_box
    parent_w = px2 - px1
    parent_h = py2 - py1
    return (
        px1 + parent_w * (cx1 / 1000.0),
        py1 + parent_h * (cy1 / 1000.0),
        px1 + parent_w * (cx2 / 1000.0),
        py1 + parent_h * (cy2 / 1000.0),
    )


def expand_zoom_box_to_aspect(box: ZoomBox, *, source_width: int, source_height: int) -> ZoomBox:
    """Expand a normalized box to the source aspect ratio without leaving bounds."""
    if source_width <= 0 or source_height <= 0:
        return box

    x1, y1, x2, y2 = box
    target_aspect = source_width / source_height
    width = x2 - x1
    height = y2 - y1
    current_aspect = width / height

    if current_aspect < target_aspect:
        new_width = height * target_aspect
        center = (x1 + x2) / 2
        x1 = center - new_width / 2
        x2 = center + new_width / 2
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if x2 > 1000:
            x1 -= x2 - 1000
            x2 = 1000.0
    else:
        new_height = width / target_aspect
        center = (y1 + y2) / 2
        y1 = center - new_height / 2
        y2 = center + new_height / 2
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if y2 > 1000:
            y1 -= y2 - 1000
            y2 = 1000.0

    return validate_zoom_box((max(0.0, x1), max(0.0, y1), min(1000.0, x2), min(1000.0, y2)))


def zoom_box_to_pixels(box: ZoomBox, *, source_width: int, source_height: int) -> tuple[int, int, int, int]:
    """Convert a 0-1000 box to ffmpeg crop pixels."""
    x1, y1, x2, y2 = box
    x = int(round(source_width * x1 / 1000.0))
    y = int(round(source_height * y1 / 1000.0))
    width = int(round(source_width * (x2 - x1) / 1000.0))
    height = int(round(source_height * (y2 - y1) / 1000.0))

    x = max(0, min(x, max(0, source_width - 1)))
    y = max(0, min(y, max(0, source_height - 1)))
    width = max(1, min(width, source_width - x))
    height = max(1, min(height, source_height - y))
    return x, y, width, height


def _zoom_filter(box: ZoomBox, *, source_width: int, source_height: int) -> str:
    x, y, width, height = zoom_box_to_pixels(
        box,
        source_width=source_width,
        source_height=source_height,
    )
    return f"crop={width}:{height}:{x}:{y},scale={source_width}:{source_height}"


def extract_zoomed_frame(
    video_path: str,
    at_s: float,
    output_path: str,
    *,
    zoom_box: ZoomBox,
    source_width: int,
    source_height: int,
) -> str:
    """Extract one cropped-and-scaled frame."""
    _require_binary("ffmpeg")
    at_s = max(0.0, float(at_s))

    src = Path(video_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{at_s:.3f}",
        "-i", str(src),
        "-vf", _zoom_filter(zoom_box, source_width=source_width, source_height=source_height),
        "-frames:v", "1",
        "-q:v", "2",
        str(dst),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg zoomed frame extraction failed")
    return str(dst)


def extract_zoomed_clip(
    video_path: str,
    start_s: float,
    end_s: float,
    output_path: str,
    *,
    zoom_box: ZoomBox,
    source_width: int,
    source_height: int,
) -> str:
    """Extract one cropped-and-scaled video clip."""
    _require_binary("ffmpeg")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.1, float(end_s))

    src = Path(video_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(src),
        "-vf", _zoom_filter(zoom_box, source_width=source_width, source_height=source_height),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg zoomed clip extraction failed")
    return str(dst)


def extract_audio(video_path: str, start_s: float, end_s: float, output_path: str) -> str:
    """Extract mono 16kHz WAV audio for a slice."""
    _require_binary("ffmpeg")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.1, float(end_s))

    src = Path(video_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg audio extraction failed")
    return str(dst)


def sample_frames(
    video_path: str,
    start_s: float,
    end_s: float,
    output_dir: str,
    *,
    max_frames: int = 8,
) -> list[str]:
    """Sample up to max_frames between start/end timestamps."""
    _require_binary("ffmpeg")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.1, float(end_s))
    duration = max(0.1, end_s - start_s)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = max_frames / duration
    pattern = out_dir / "frame_%04d.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", video_path,
        "-vf", f"fps={fps:.4f}",
        "-q:v", "2",
        str(pattern),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg frame sampling failed")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) > max_frames:
        for extra in frames[max_frames:]:
            extra.unlink(missing_ok=True)
        frames = frames[:max_frames]

    return [str(path) for path in frames]
