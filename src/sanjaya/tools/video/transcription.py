"""Video transcript helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...settings import get_settings


class TranscriptionError(RuntimeError):
    """Raised when transcript preparation fails."""


_MAX_UPLOAD_MB = 24.0
_MAX_TRANSCRIPTION_RETRY_WAIT_S = 60.0
_INITIAL_TRANSCRIPTION_RETRY_DELAY_S = 1.0


@dataclass
class TranscriptPreparationResult:
    """Result of preparing a transcript for a video run."""

    transcript: dict[str, Any]
    transcript_path: str | None
    generated: bool
    source: str


@dataclass
class SubtitlePreparationResult:
    """Legacy result of transcript sidecar lookup/generation."""

    subtitle_path: str | None
    generated: bool
    source: str | None
    error: str | None = None


def _segment_value(segment: Any, *names: str) -> Any:
    for name in names:
        value = getattr(segment, name, None)
        if value is not None:
            return value
        if isinstance(segment, dict) and name in segment:
            return segment[name]
    return None


def _normalize_segments(
    raw_segments: list[Any],
    *,
    offset_s: float = 0.0,
    speaker_prefix: str | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in raw_segments:
        start = _segment_value(segment, "start_s", "start", "start_time", "from")
        end = _segment_value(segment, "end_s", "end", "end_time", "to")
        text = _segment_value(segment, "text", "subtitle", "content")
        speaker = _segment_value(segment, "speaker")

        try:
            start_s = float(start) + offset_s
            end_s = float(end) + offset_s
        except (TypeError, ValueError):
            continue

        if end_s <= start_s:
            continue

        text_value = str(text or "").strip()
        if not text_value:
            continue

        entry: dict[str, Any] = {
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "text": text_value,
        }
        if speaker is not None:
            speaker_value = str(speaker)
            entry["speaker"] = f"{speaker_prefix}{speaker_value}" if speaker_prefix else speaker_value

        normalized.append(entry)

    return normalized


def _build_transcript(segments: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(segments, key=lambda item: (item["start_s"], item["end_s"]))
    normalized_segments = []
    for index, segment in enumerate(ordered):
        entry = dict(segment)
        entry["index"] = index
        normalized_segments.append(entry)

    return {
        "text": " ".join(segment["text"] for segment in normalized_segments).strip(),
        "segments": normalized_segments,
        "metadata": {
            **metadata,
            "segment_count": len(normalized_segments),
        },
    }


def _write_sidecar(path: Path, transcript: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return str(path)


def _extract_payload_segments(payload: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], {}

    if not isinstance(payload, dict):
        return [], {}

    for key in ("segments", "subtitles", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value, dict(payload.get("metadata") or {})
    return [], dict(payload.get("metadata") or {})


def load_transcript(path: str) -> dict[str, Any]:
    """Load a transcript sidecar and normalize it into the REPL transcript shape."""
    sidecar = Path(path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    raw_segments, metadata = _extract_payload_segments(payload)
    segments = _normalize_segments(raw_segments)
    metadata.setdefault("source", "existing-sidecar")
    metadata.setdefault("transcript_path", str(sidecar))
    return _build_transcript(segments, metadata)


def _run_media_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptionError(result.stderr.strip() or f"media command failed: {' '.join(cmd)}")


def _retry_transcription_step[T](operation: Callable[[], T]) -> T:
    waited_s = 0.0
    delay_s = _INITIAL_TRANSCRIPTION_RETRY_DELAY_S

    while True:
        try:
            return operation()
        except Exception:
            remaining_s = _MAX_TRANSCRIPTION_RETRY_WAIT_S - waited_s
            if remaining_s <= 0:
                raise

            sleep_s = min(delay_s, remaining_s)
            time.sleep(sleep_s)
            waited_s += sleep_s
            delay_s *= 2


def _probe_duration_s(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise TranscriptionError("ffprobe is required for automatic transcription")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise TranscriptionError(result.stderr.strip() or f"ffprobe failed for {path}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise TranscriptionError(f"Could not read media duration for {path}") from exc


def _extract_upload_audio(
    *,
    video_path: Path,
    output_path: Path,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("ffmpeg is required for automatic transcription")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start_s is not None:
        cmd.extend(["-ss", f"{start_s:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_s is not None:
        cmd.extend(["-t", f"{duration_s:.3f}"])
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", str(output_path)])
    _run_media_command(cmd)


def _audio_upload_chunks(video_path: Path, tmp_path: Path) -> list[tuple[Path, float]]:
    audio_path = tmp_path / "audio.mp3"
    _extract_upload_audio(video_path=video_path, output_path=audio_path)
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb <= _MAX_UPLOAD_MB:
        return [(audio_path, 0.0)]

    duration_s = _probe_duration_s(video_path)
    chunk_count = max(2, math.ceil(size_mb / _MAX_UPLOAD_MB))
    chunk_duration_s = duration_s / chunk_count
    chunks: list[tuple[Path, float]] = []
    for index in range(chunk_count):
        start_s = index * chunk_duration_s
        end_s = min(duration_s, start_s + chunk_duration_s)
        if end_s <= start_s:
            continue
        chunk_path = tmp_path / f"audio_{index:03d}.mp3"
        _extract_upload_audio(
            video_path=video_path,
            output_path=chunk_path,
            start_s=start_s,
            duration_s=end_s - start_s,
        )
        if chunk_path.stat().st_size / (1024 * 1024) > 25:
            raise TranscriptionError(f"Audio chunk is still too large for upload: {chunk_path.name}")
        chunks.append((chunk_path, start_s))
    return chunks


def transcribe_with_whisper_local(
    *,
    video_path: str,
    output_path: str,
    model: str = "base",
    language: str | None = None,
) -> str:
    """Generate transcript JSON using local whisper CLI with timestamps."""
    if shutil.which("whisper") is None:
        raise TranscriptionError("Local whisper CLI not found. Install openai-whisper to use local transcription.")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    out = Path(output_path)

    with tempfile.TemporaryDirectory(prefix="sanjaya-whisper-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            "whisper",
            str(src),
            "--model", model,
            "--task", "transcribe",
            "--output_format", "json",
            "--output_dir", str(tmp_path),
            "--fp16", "False",
        ]
        if language:
            cmd.extend(["--language", language])

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise TranscriptionError(result.stderr.strip() or "whisper transcription failed")

        whisper_json = tmp_path / f"{src.stem}.json"
        if not whisper_json.exists():
            raise TranscriptionError(f"whisper did not produce expected output: {whisper_json}")

        payload = json.loads(whisper_json.read_text(encoding="utf-8"))
        segments = _normalize_segments(payload.get("segments", []))

        transcript = _build_transcript(
            segments,
            {
                "source": "whisper-local",
                "model": model,
                "language": language or "auto",
                "video_path": str(src),
            },
        )
        return _write_sidecar(out, transcript)


def _is_diarize_model(model: str) -> bool:
    return "diarize" in model.lower()


def _is_gpt_transcribe_model(model: str) -> bool:
    return model.startswith("gpt-4o") and "transcribe" in model.lower()


def _transcribe_audio_chunk(
    *,
    client: Any,
    audio_path: Path,
    model: str,
    language: str | None,
) -> list[Any]:
    create_kwargs: dict[str, Any] = {
        "model": model,
    }
    if language:
        create_kwargs["language"] = language

    if _is_diarize_model(model):
        create_kwargs["response_format"] = "diarized_json"
        create_kwargs["chunking_strategy"] = "auto"
    elif model == "whisper-1":
        create_kwargs["response_format"] = "verbose_json"
        create_kwargs["timestamp_granularities"] = ["segment"]
    elif _is_gpt_transcribe_model(model):
        create_kwargs["response_format"] = "json"
    else:
        create_kwargs["response_format"] = "verbose_json"
        create_kwargs["timestamp_granularities"] = ["segment"]

    with audio_path.open("rb") as audio_file:
        create_kwargs["file"] = audio_file
        response = client.audio.transcriptions.create(**create_kwargs)

    raw_segments = getattr(response, "segments", None)
    if raw_segments is None and isinstance(response, dict):
        raw_segments = response.get("segments")
    return list(raw_segments or [])


def transcribe_with_openai_api(
    *,
    video_path: str,
    output_path: str,
    model: str = "whisper-1",
    language: str | None = None,
    api_key: str | None = None,
) -> str:
    """Generate transcript JSON using OpenAI transcription API with segment timestamps."""
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    key = api_key or os.getenv("OPENAI_API_KEY") or get_settings().openai_api_key
    if not key:
        raise TranscriptionError("OPENAI_API_KEY is required for API transcription")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise TranscriptionError("openai package not available for API transcription") from exc

    client = OpenAI(api_key=key)

    with tempfile.TemporaryDirectory(prefix="sanjaya-transcribe-") as tmp_dir:
        chunks = _audio_upload_chunks(src, Path(tmp_dir))
        multiple_chunks = len(chunks) > 1

        def _transcribe_chunks() -> list[dict[str, Any]]:
            segments: list[dict[str, Any]] = []
            for chunk_index, (audio_path, offset_s) in enumerate(chunks):
                raw_segments = _transcribe_audio_chunk(
                    client=client,
                    audio_path=audio_path,
                    model=model,
                    language=language,
                )
                speaker_prefix = f"chunk_{chunk_index}_" if multiple_chunks else None
                segments.extend(
                    _normalize_segments(raw_segments, offset_s=offset_s, speaker_prefix=speaker_prefix)
                )

            return segments

        segments = _retry_transcription_step(_transcribe_chunks)

    speaker_label_scope = "per_chunk" if multiple_chunks and _is_diarize_model(model) else "global"
    if not _is_diarize_model(model):
        speaker_label_scope = "none"

    transcript = _build_transcript(
        segments,
        {
            "source": "openai-api",
            "model": model,
            "language": language or "auto",
            "video_path": str(src),
            "speaker_label_scope": speaker_label_scope,
        },
    )
    out = Path(output_path)
    return _write_sidecar(out, transcript)


def _transcript_model_slug(model: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in model.strip())
    return "-".join(part for part in slug.split("-") if part) or "model"


def _transcript_sidecar_name(src: Path, model: str) -> str:
    return f"{src.stem}_{_transcript_model_slug(model)}_transcript.json"


def _inferred_transcript_candidates(src: Path, output_dir: str | None = None, model: str = "whisper-1") -> list[Path]:
    filename = _transcript_sidecar_name(src, model)
    candidates = [
        src.with_name(filename),
        src.parent / "meta" / filename,
        src.parent.parent / "meta" / filename,
    ]
    if output_dir is not None:
        candidates.append(Path(output_dir) / filename)
    return candidates


def prepare_video_transcript(
    *,
    video_path: str,
    explicit_transcript_path: str | None = None,
    output_dir: str | None = None,
    api_model: str = "whisper-1",
    language: str | None = None,
) -> TranscriptPreparationResult:
    """Load or generate a timestamped transcript for a video run."""
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    explicit_path = Path(explicit_transcript_path) if explicit_transcript_path else None
    if explicit_path and explicit_path.exists():
        return TranscriptPreparationResult(
            transcript=load_transcript(str(explicit_path)),
            transcript_path=str(explicit_path),
            generated=False,
            source="existing",
        )

    for candidate in _inferred_transcript_candidates(src, output_dir, api_model):
        if candidate.exists():
            return TranscriptPreparationResult(
                transcript=load_transcript(str(candidate)),
                transcript_path=str(candidate),
                generated=False,
                source="existing",
            )

    sidecar_name = _transcript_sidecar_name(src, api_model)
    if output_dir is None:
        target = explicit_path or (src.parent / "meta" / sidecar_name)
    else:
        target = explicit_path or (Path(output_dir) / sidecar_name)

    generated_path = transcribe_with_openai_api(
        video_path=video_path,
        output_path=str(target),
        model=api_model,
        language=language,
    )
    return TranscriptPreparationResult(
        transcript=load_transcript(generated_path),
        transcript_path=generated_path,
        generated=True,
        source="openai-api",
    )


def ensure_subtitle_sidecar(
    *,
    video_path: str,
    explicit_subtitle_path: str | None = None,
    mode: str = "auto",
    output_dir: str | None = None,
    local_model: str = "base",
    api_model: str = "whisper-1",
    language: str | None = None,
) -> SubtitlePreparationResult:
    """Find or generate subtitle sidecar using local or API transcription."""
    mode_normalized = mode.strip().lower()
    if mode_normalized not in {"auto", "local", "api", "none"}:
        raise ValueError("subtitle mode must be one of: auto, local, api, none")

    explicit_path = Path(explicit_subtitle_path) if explicit_subtitle_path else None
    if explicit_path and explicit_path.exists():
        return SubtitlePreparationResult(
            subtitle_path=str(explicit_path),
            generated=False,
            source="existing",
        )

    src = Path(video_path)
    sidecar_name = _transcript_sidecar_name(src, api_model)

    inferred_candidates = [
        src.with_name(sidecar_name),
        src.parent / "meta" / sidecar_name,
        src.parent.parent / "meta" / sidecar_name,
    ]
    if output_dir is not None:
        inferred_candidates.append(Path(output_dir) / sidecar_name)

    for candidate in inferred_candidates:
        if candidate.exists():
            return SubtitlePreparationResult(
                subtitle_path=str(candidate),
                generated=False,
                source="existing",
            )

    if mode_normalized == "none":
        return SubtitlePreparationResult(subtitle_path=None, generated=False, source=None)

    if output_dir is None:
        # Default: write sidecar next to the video in a meta/ subdirectory
        target = explicit_path or (src.parent / "meta" / sidecar_name)
    else:
        target = explicit_path or (Path(output_dir) / sidecar_name)

    errors: list[str] = []

    def _try_local() -> str:
        return transcribe_with_whisper_local(
            video_path=video_path,
            output_path=str(target),
            model=local_model,
            language=language,
        )

    def _try_api() -> str:
        return transcribe_with_openai_api(
            video_path=video_path,
            output_path=str(target),
            model=api_model,
            language=language,
        )

    order: list[tuple[str, Any]] = []
    if mode_normalized == "local":
        order = [("whisper-local", _try_local)]
    elif mode_normalized == "api":
        order = [("openai-api", _try_api)]
    else:
        order = [("whisper-local", _try_local), ("openai-api", _try_api)]

    for source, producer in order:
        try:
            generated_path = producer()
            return SubtitlePreparationResult(
                subtitle_path=generated_path,
                generated=True,
                source=source,
            )
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    return SubtitlePreparationResult(
        subtitle_path=None,
        generated=False,
        source=None,
        error="; ".join(errors),
    )
