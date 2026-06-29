"""Adapter that lets the external MMOU benchmark call Sanjaya as a model."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sanjaya import Agent
from sanjaya.answer import Answer
from sanjaya.model_defaults import DEFAULT_AUDIO_MODEL, DEFAULT_ROOT_MODEL, DEFAULT_SUB_MODEL, DEFAULT_VISION_MODEL
from sanjaya.prompts import PromptConfig
from sanjaya.tools.video import VideoToolkit

ANSWER_PATTERN = re.compile(r"\b([A-J])\b", re.IGNORECASE)
_LEAKY_PROMPT_PREFIXES = ("question id:", "domain:", "evidence window:")


def _generation_response_class() -> Any:
    try:
        from videobench.models.base import GenerationResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "videobench is required for the MMOU adapter. Add the benchmarks repo's src directory to PYTHONPATH."
        ) from exc
    return GenerationResponse


def _answer_schema() -> dict[str, Any]:
    return {
        "answer_type": "mmou_multiple_choice",
        "fields": {
            "answer": {
                "type": "str",
                "required": True,
                "description": "Exactly one uppercase answer letter from A through J.",
            },
        },
    }


def _sanitize_mmou_prompt(prompt: Any) -> str:
    """Remove MMOU benchmark metadata before the prompt reaches Sanjaya."""
    raw = str(prompt or "").strip()
    if not raw:
        return raw

    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("answer the multiple-choice question"):
            lines = lines[index:]
            break

    cleaned_lines: list[str] = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized.startswith(_LEAKY_PROMPT_PREFIXES):
            continue
        if "downstream multiple-choice qa system" in normalized:
            continue
        if normalized.startswith("whole video summary:"):
            continue
        if normalized.startswith("use the summary as high-level context"):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\bMMOU\b", "video", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbenchmark\b", "task", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bevaluator\b", "assistant", cleaned, flags=re.IGNORECASE)
    return cleaned


def _slug(value: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "-" for ch in value]
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "sample"


def _request_cache_key(request: Any, adapter_name: str) -> str:
    cache_key = getattr(request, "cache_key", None)
    if callable(cache_key):
        return str(cache_key(adapter_name))
    payload = {
        "adapter": adapter_name,
        "model": getattr(request, "model", ""),
        "prompt": getattr(request, "prompt", ""),
        "metadata": getattr(request, "metadata", {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _local_video_path(request: Any) -> Path:
    for media in getattr(request, "media", []) or []:
        path = getattr(media, "path", None)
        if path:
            candidate = Path(path)
            if not candidate.exists():
                raise FileNotFoundError(f"MMOU video path does not exist: {candidate}")
            return candidate
    raise ValueError("Sanjaya MMOU adapter requires a local video path. Use videobench URL mode, not raw URLs.")


def _parse_answer_letter(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Empty model response.")
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "answer" in payload:
            answer = str(payload["answer"]).strip().upper()
            if re.fullmatch(r"[A-J]", answer):
                return answer
    except json.JSONDecodeError:
        pass

    match = ANSWER_PATTERN.search(cleaned)
    if not match:
        raise ValueError("Could not parse an answer letter from the model response.")
    return match.group(1).upper()


def _response_text(answer: Answer) -> tuple[str, dict[str, str] | None]:
    candidates: list[str] = []
    if isinstance(answer.data, dict) and answer.data.get("answer") is not None:
        candidates.append(str(answer.data["answer"]))
        candidates.append(json.dumps({"answer": answer.data["answer"]}))
    candidates.append(answer.text)

    for candidate in candidates:
        try:
            letter = _parse_answer_letter(candidate)
        except ValueError:
            continue
        payload = {"answer": letter}
        return json.dumps(payload), payload

    return answer.text.strip(), None


def _answer_raw(answer: Any) -> dict[str, Any]:
    if hasattr(answer, "model_dump"):
        try:
            dumped = answer.model_dump(mode="json")
        except TypeError:
            dumped = answer.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _answer_usage(answer: Answer) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    if answer.input_tokens is not None:
        usage["input_tokens"] = answer.input_tokens
    if answer.output_tokens is not None:
        usage["output_tokens"] = answer.output_tokens
    if answer.cost_usd is not None:
        usage["cost_usd"] = answer.cost_usd
    if answer.wall_time_s is not None:
        usage["wall_time_s"] = answer.wall_time_s
    usage["iterations"] = answer.iterations
    usage["evidence_count"] = len(answer.evidence)
    return usage


@dataclass(slots=True)
class SanjayaMMOUAdapter:
    """Small videobench-compatible adapter for MMOU prediction."""

    name: str = "sanjaya-rlm"
    root_model: str = DEFAULT_ROOT_MODEL
    sub_model: str = DEFAULT_SUB_MODEL
    recursive_model: str = DEFAULT_ROOT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    audio_model: str | None = DEFAULT_AUDIO_MODEL
    max_iterations: int = 8
    max_depth: int = 2
    max_budget_usd: float | None = None
    max_timeout_s: float | None = None
    max_span_s: float = 120.0
    tracing: bool = True
    keep_artifacts: bool = False
    artifacts_root: str | None = None
    agent_factory: Callable[..., Agent] = Agent
    agent_callback: Callable[[Agent, dict[str, Any]], None] | None = None

    def generate(self, request: Any) -> Any:
        response_cls = _generation_response_class()
        video_path = _local_video_path(request)
        metadata = dict(getattr(request, "metadata", {}) or {})
        question_id = str(metadata.get("question_id") or "mmou")
        digest = hashlib.sha1(
            f"{question_id}|{getattr(request, 'prompt', '')}|{video_path.name}".encode("utf-8")
        ).hexdigest()[:8]
        run_id = f"mmou-{_slug(question_id)}-{digest}"

        if self.keep_artifacts:
            workspace_context = nullcontext(str(Path(self.artifacts_root or "sanjaya_artifacts/mmou")))
        else:
            workspace_context = tempfile.TemporaryDirectory(prefix="sanjaya-video-")

        with workspace_context as workspace_dir:
            Path(workspace_dir).mkdir(parents=True, exist_ok=True)
            toolkit = VideoToolkit(workspace_dir=str(workspace_dir), max_span_s=self.max_span_s)
            agent = self.agent_factory(
                model=self.root_model,
                sub_model=self.sub_model,
                recursive_model=self.recursive_model,
                vision_model=self.vision_model,
                audio_model=self.audio_model,
                caption_model=None,
                fallback_model=None,
                critic_model=None,
                prompts=PromptConfig(answer_schema=_answer_schema()),
                max_iterations=self.max_iterations,
                max_depth=self.max_depth,
                max_budget_usd=self.max_budget_usd,
                max_timeout_s=self.max_timeout_s,
                tracing=self.tracing,
            ).use(toolkit)
            if self.agent_callback is not None:
                self.agent_callback(
                    agent,
                    {
                        "run_id": run_id,
                        "question_id": question_id,
                        "workspace_dir": str(workspace_dir),
                    },
                )
            answer = agent.ask(_sanitize_mmou_prompt(getattr(request, "prompt", "")), video=str(video_path))

        text, parsed_json = _response_text(answer)
        return response_cls(
            model=str(getattr(request, "model", self.name)),
            text=text,
            parsed_json=parsed_json,
            raw={"sanjaya": _answer_raw(answer)},
            usage=_answer_usage(answer),
            cache_key=_request_cache_key(request, self.name),
            cached=False,
        )
