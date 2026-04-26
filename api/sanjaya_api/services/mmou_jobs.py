"""Background service for MMOU benchmark jobs."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if "sanjaya" in sys.modules:
    package_path = str(SRC_DIR / "sanjaya")
    sanjaya_package = sys.modules["sanjaya"]
    search_path = getattr(sanjaya_package, "__path__", None)
    if search_path is not None and package_path not in search_path:
        search_path.insert(0, package_path)

from sanjaya.benchmarks import SanjayaMMOUAdapter
from sanjaya.model_defaults import DEFAULT_AUDIO_MODEL, DEFAULT_ROOT_MODEL, DEFAULT_SUB_MODEL, DEFAULT_VISION_MODEL

from sanjaya_api.models import (
    MMOUCatalogResponse,
    MMOUEvaluationSummary,
    MMOUJobCreateRequest,
    MMOUJobSummary,
    MMOUQuestionEvaluationSummary,
    MMOUQuestionStatus,
)
from sanjaya_api.trace_events import normalize_trace_event

ARTIFACTS_DIR = PROJECT_ROOT / "sanjaya_artifacts"
DEFAULT_MMOU_JOBS_DIR = ARTIFACTS_DIR / "mmou_jobs"
DEFAULT_BENCHMARKS_DIR = Path(os.getenv("SANJAYA_BENCHMARKS_DIR", "/Users/lsteno/Developer/GitHub/benchmarks"))
MODEL_NAME = "sanjaya-rlm"


@lru_cache(maxsize=4)
def _load_mmou_benchmark_api_cached(benchmarks_dir: str) -> SimpleNamespace:
    root = Path(benchmarks_dir)
    src_dir = root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Benchmark repo src directory not found: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from videobench.benchmarks.mmou import (
        MMOU_DATASET_FILE,
        MMOU_EVAL_SPACE,
        MMOUSample,
        build_mmou_prompt,
        download_mmou_metadata,
        download_remote_video,
        evaluate_submission_with_api,
        export_submission_from_predictions,
        load_mmou_dataset,
        mmou_domain_counts,
        parse_answer_letter,
    )
    from videobench.config import load_config
    from videobench.models.base import GenerationRequest, MediaInput, MediaType

    return SimpleNamespace(
        MMOU_DATASET_FILE=MMOU_DATASET_FILE,
        MMOUSample=MMOUSample,
        build_mmou_prompt=build_mmou_prompt,
        download_mmou_metadata=download_mmou_metadata,
        download_remote_video=download_remote_video,
        evaluate_submission_with_api=evaluate_submission_with_api,
        export_submission_from_predictions=export_submission_from_predictions,
        GenerationRequest=GenerationRequest,
        load_config=load_config,
        load_mmou_dataset=load_mmou_dataset,
        MediaInput=MediaInput,
        MediaType=MediaType,
        mmou_domain_counts=mmou_domain_counts,
        MMOU_EVAL_SPACE=MMOU_EVAL_SPACE,
        parse_answer_letter=parse_answer_letter,
    )


def load_mmou_benchmark_api(benchmarks_dir: str | None = None) -> SimpleNamespace:
    """Load the external MMOU benchmark helpers."""
    return _load_mmou_benchmark_api_cached(str(Path(benchmarks_dir) if benchmarks_dir else DEFAULT_BENCHMARKS_DIR))


def _resolve_output_dir(output_dir: str | None) -> Path:
    if not output_dir:
        return DEFAULT_MMOU_JOBS_DIR
    candidate = Path(output_dir)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _resolve_benchmarks_dir(benchmarks_dir: str | None) -> Path:
    candidate = Path(benchmarks_dir) if benchmarks_dir else DEFAULT_BENCHMARKS_DIR
    return candidate.expanduser().resolve()


def _resolve_dataset_path(api: SimpleNamespace, request: MMOUJobCreateRequest | None = None) -> Path:
    if request is not None and request.dataset_file:
        candidate = Path(request.dataset_file).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    else:
        benchmarks_dir = _resolve_benchmarks_dir(None if request is None else request.benchmarks_dir)
        cfg = api.load_config(None, cwd=benchmarks_dir)
        candidate = cfg.storage.data_dir / "mmou" / api.MMOU_DATASET_FILE

    if not candidate.exists():
        api.download_mmou_metadata(candidate.parent, include_captions=False)
    if not candidate.exists():
        raise FileNotFoundError(f"MMOU dataset file not found: {candidate}")
    return candidate


def _sample_to_payload(sample: Any) -> dict[str, Any]:
    return {
        "question_id": sample.question_id,
        "question": sample.question,
        "options": dict(sample.options),
        "video_url": sample.video_url,
        "domain": sample.domain,
        "subdomain": sample.subdomain,
        "question_type": list(sample.question_type),
        "start_time": sample.start_time,
        "end_time": sample.end_time,
        "video_duration": sample.video_duration,
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned[:120] or "question"


def _dedupe_question_ids(question_ids: list[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_id in question_ids or []:
        question_id = str(raw_id).strip()
        if not question_id or question_id in seen:
            continue
        seen.add(question_id)
        ordered.append(question_id)
    return ordered


def select_mmou_samples_for_ui(
    samples: list[Any],
    *,
    limit: int,
    stratified: bool,
    domains: list[str] | None = None,
    question_ids: list[str] | None = None,
) -> tuple[list[Any], str]:
    """Select samples with exact round-robin stratification for UI runs."""
    ids = _dedupe_question_ids(question_ids)
    if ids:
        by_id = {sample.question_id: sample for sample in samples}
        missing = [question_id for question_id in ids if question_id not in by_id]
        if missing:
            raise ValueError(f"Unknown MMOU question ids: {missing[:20]}")
        return [by_id[question_id] for question_id in ids[:limit]], "question_ids"

    selected = list(samples)
    if domains:
        wanted = {domain.strip().lower() for domain in domains if domain.strip()}
        selected = [sample for sample in selected if sample.domain.strip().lower() in wanted]

    if not stratified:
        return selected[:limit], "dataset"

    by_domain: dict[str, list[Any]] = {}
    domain_order: list[str] = []
    for sample in selected:
        domain = sample.domain.strip()
        if domain not in by_domain:
            by_domain[domain] = []
            domain_order.append(domain)
        by_domain[domain].append(sample)

    balanced: list[Any] = []
    offset = 0
    while len(balanced) < limit:
        added = False
        for domain in domain_order:
            bucket = by_domain[domain]
            if offset < len(bucket):
                balanced.append(bucket[offset])
                added = True
                if len(balanced) >= limit:
                    break
        if not added:
            break
        offset += 1
    return balanced, "dataset"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, default=str) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def _is_completed_prediction(row: dict[str, Any]) -> bool:
    answer = row.get("answer")
    return bool(row.get("question_id")) and isinstance(answer, str) and bool(answer.strip())


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _find_evaluator_overall_row(evaluation: dict[str, Any]) -> dict[str, Any] | None:
    duration = evaluation.get("duration_breakdown")
    if not isinstance(duration, dict):
        return None
    headers = [str(header).strip().lower() for header in duration.get("headers") or []]
    rows = duration.get("data") or []
    if not headers or not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        if str(row[0]).strip().lower() != "overall":
            continue
        values = {headers[index]: row[index] for index in range(min(len(headers), len(row)))}
        accuracy = _coerce_float(values.get("answered accuracy (%)"))
        correct = _coerce_int(values.get("correct"))
        answered = _coerce_int(values.get("answered"))
        if accuracy is None or correct is None or answered is None:
            return None
        return {
            "answered_accuracy_pct": accuracy,
            "correct": correct,
            "answered": answered,
        }
    return None


def _parse_evaluator_markdown(evaluation: dict[str, Any]) -> dict[str, Any] | None:
    for markdown in evaluation.get("markdown_outputs") or []:
        if not isinstance(markdown, str) or "Answered accuracy" not in markdown:
            continue
        match = re.search(
            r"Answered accuracy:\s*`?([0-9.]+)%`?\s*\(`?(\d+)\s*/\s*(\d+)`?\)",
            markdown,
            flags=re.IGNORECASE,
        )
        if match:
            return {
                "answered_accuracy_pct": float(match.group(1)),
                "correct": int(match.group(2)),
                "answered": int(match.group(3)),
            }
    return None


def _compact_evaluation_summary(evaluation: dict[str, Any], *, submission_rows: int) -> MMOUEvaluationSummary:
    parsed = _find_evaluator_overall_row(evaluation) or _parse_evaluator_markdown(evaluation)
    if parsed is None:
        raise ValueError("Could not parse answered accuracy from MMOU evaluator response.")
    return MMOUEvaluationSummary(
        answered_accuracy_pct=float(parsed["answered_accuracy_pct"]),
        correct=int(parsed["correct"]),
        answered=int(parsed["answered"]),
        evaluated_at=str(evaluation.get("evaluated_at") or datetime.now(timezone.utc).isoformat()),
        submission_rows=submission_rows,
    )


def _compact_question_evaluation_summary(
    question_id: str,
    answer: str,
    evaluation: dict[str, Any],
    *,
    submission_rows: int,
) -> MMOUQuestionEvaluationSummary:
    parsed = _find_evaluator_overall_row(evaluation) or _parse_evaluator_markdown(evaluation)
    if parsed is None:
        raise ValueError("Could not parse answered accuracy from MMOU evaluator response.")
    correct = int(parsed["correct"])
    answered = int(parsed["answered"])
    if answered < 1:
        raise ValueError("MMOU evaluator did not score the answered question.")
    return MMOUQuestionEvaluationSummary(
        question_id=question_id,
        answer=answer,
        correct=correct > 0,
        answered_accuracy_pct=float(parsed["answered_accuracy_pct"]),
        evaluated_at=str(evaluation.get("evaluated_at") or datetime.now(timezone.utc).isoformat()),
        submission_rows=submission_rows,
    )


def _prediction_row(
    question: "MMOUQuestionRecord",
    *,
    video_ref: str,
    answer: str | None,
    raw_text: str | None,
    cache_key: str | None,
    cached: bool,
    usage: dict[str, Any],
    parse_error: str | None,
    attempts: int,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": question.question_id,
        "question": question.question,
        "domain": question.domain,
        "subdomain": question.subdomain,
        "question_type": list(question.question_type),
        "video_ref": video_ref,
        "answer": answer,
        "raw_text": raw_text,
        "cache_key": cache_key,
        "cached": cached,
        "usage": usage,
        "parse_error": parse_error,
        "attempts": attempts,
        "clip_mode": "full",
        "clip_start_seconds": None,
        "clip_end_seconds": None,
        "clip_duration_seconds": None,
        "video_summary_context_used": False,
        "video_summary_text": None,
        "video_summary_cache_key": None,
        "video_summary_cached": False,
        "video_summary_usage": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        row["error"] = error
        row["skipped"] = True
    return row


@dataclass
class MMOUQuestionRecord:
    question_id: str
    question: str
    options: dict[str, str]
    video_url: str
    domain: str
    subdomain: str
    question_type: list[str]
    start_time: str
    end_time: str
    video_duration: float | None
    status: str = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    run_id: str | None = None
    result_path: str | None = None
    trace_path: str | None = None
    answer: str | None = None
    raw_text: str | None = None
    parse_error: str | None = None
    attempts: int | None = None
    iterations: int | None = None
    cost_usd: float | None = None
    wall_time_s: float | None = None
    error: str | None = None
    tracer: Any | None = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    prediction_row: dict[str, Any] | None = None


@dataclass
class MMOUJobRecord:
    job_id: str
    request: MMOUJobCreateRequest
    run_name: str
    output_dir: str
    job_dir: str
    models: dict[str, str | None]
    question_ids: list[str]
    questions: dict[str, MMOUQuestionRecord]
    selection_source: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested_at: float | None = None
    stop_reason: str | None = None
    stdout_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    revision: int = 0
    thread: threading.Thread | None = None


class MMOUBenchmarkJobService:
    """Manages MMOU benchmark jobs with per-question live traces and resume."""

    def __init__(self, output_root: Path = DEFAULT_MMOU_JOBS_DIR) -> None:
        self._jobs: dict[str, MMOUJobRecord] = {}
        self._lock = threading.RLock()
        self._output_root = output_root
        self._hydrate_persisted_jobs()

    def get_catalog(self) -> MMOUCatalogResponse:
        api = load_mmou_benchmark_api()
        dataset_path = _resolve_dataset_path(api)
        samples = api.load_mmou_dataset(dataset_path)
        return MMOUCatalogResponse(
            total_questions=len(samples),
            domain_counts=api.mmou_domain_counts(samples),
            defaults={
                "limit": 10,
                "stratified": True,
                "workers": 1,
                "max_iterations": 20,
                "max_depth": 2,
                "max_budget_usd": None,
                "max_timeout_s": None,
                "output_dir": str(self._output_root),
                "keep_artifacts": False,
                "models": {
                    "root": DEFAULT_ROOT_MODEL,
                    "sub": DEFAULT_SUB_MODEL,
                    "recursive": DEFAULT_ROOT_MODEL,
                    "vision": DEFAULT_VISION_MODEL,
                    "audio": DEFAULT_AUDIO_MODEL,
                },
            },
        )

    def start_job(self, request: MMOUJobCreateRequest) -> MMOUJobSummary:
        api = load_mmou_benchmark_api(request.benchmarks_dir)
        dataset_path = _resolve_dataset_path(api, request)
        samples = api.load_mmou_dataset(dataset_path)
        selected, selection_source = select_mmou_samples_for_ui(
            samples,
            limit=request.limit,
            stratified=request.stratified,
            domains=request.domains,
            question_ids=request.question_ids,
        )
        if not selected:
            raise ValueError("No MMOU questions matched the requested selection.")

        output_dir = self._resolve_job_output_dir(request.output_dir)
        run_name = request.run_name or f"mmou_{time.strftime('%Y%m%d_%H%M%S')}"
        job_id = f"mmou_job_{uuid4().hex[:12]}"
        job_dir = output_dir / job_id
        self._ensure_layout(job_dir)
        models = {
            "root": DEFAULT_ROOT_MODEL,
            "sub": DEFAULT_SUB_MODEL,
            "recursive": DEFAULT_ROOT_MODEL,
            "vision": DEFAULT_VISION_MODEL,
            "audio": DEFAULT_AUDIO_MODEL,
        }

        question_records = {
            sample.question_id: MMOUQuestionRecord(**_sample_to_payload(sample))
            for sample in selected
        }
        record = MMOUJobRecord(
            job_id=job_id,
            request=request,
            run_name=run_name,
            output_dir=str(output_dir),
            job_dir=str(job_dir),
            models=models,
            question_ids=[sample.question_id for sample in selected],
            questions=question_records,
            selection_source=selection_source,
        )

        with self._lock:
            self._jobs[job_id] = record
            self._touch(record, stdout=f"Queued {len(selected)} MMOU question(s) for run {run_name}.")
            self._persist_record(record)
            self._register_job_dir(Path(record.job_dir))

        thread = threading.Thread(target=self._run_job, args=(record,), daemon=True)
        record.thread = thread
        thread.start()
        return self._serialize_job(record)

    def list_jobs(self) -> list[MMOUJobSummary]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._serialize_job(job) for job in jobs]

    def get_job(self, job_id: str) -> MMOUJobSummary | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return self._serialize_job(record) if record else None

    def get_job_record(self, job_id: str) -> MMOUJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def request_stop(self, job_id: str, *, reason: str = "Stop requested from dashboard") -> MMOUJobSummary | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.status in ("complete", "error", "stopped", "interrupted"):
                return self._serialize_job(record)
            if record.stop_requested_at is None:
                record.stop_requested_at = time.time()
                record.stop_reason = reason
                record.status = "stopping"
                self._touch(record, stdout=reason)
                self._persist_record(record)
            return self._serialize_job(record)

    def resume_job(self, job_id: str) -> MMOUJobSummary | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.status in ("pending", "running", "stopping"):
                return self._serialize_job(record)
            if record.thread is not None and record.thread.is_alive():
                return self._serialize_job(record)

            completed = self._completed_question_ids(record)
            has_work = False
            for question_id in record.question_ids:
                question = record.questions[question_id]
                if question_id in completed:
                    question.status = "complete"
                    continue
                has_work = True
                question.status = "pending"
                question.started_at = None
                question.finished_at = None
                question.error = None
                question.tracer = None
            if not has_work:
                record.status = "complete"
                record.finished_at = time.time()
                self._touch(record, stdout="No incomplete MMOU questions remain.")
                self._persist_record(record)
                return self._serialize_job(record)

            record.status = "pending"
            record.finished_at = None
            record.stop_requested_at = None
            record.stop_reason = None
            self._touch(record, stdout=f"Resuming MMOU job {record.job_id}.")
            self._persist_record(record)

            thread = threading.Thread(target=self._run_job, args=(record,), daemon=True)
            record.thread = thread
            thread.start()
            return self._serialize_job(record)

    def get_question_trace(self, job_id: str, question_id: str) -> tuple[str | None, list[dict[str, Any]]] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            question = record.questions.get(question_id)
            if question is None:
                return None
            if question.tracer is not None:
                events = [self._normalize_event(raw) for raw in question.tracer.events]
            elif question.trace_events:
                events = list(question.trace_events)
            elif question.trace_path:
                payload = _read_json(Path(question.trace_path), {})
                loaded = payload.get("events") if isinstance(payload, dict) else None
                events = loaded if isinstance(loaded, list) else []
            else:
                events = []
            return question.run_id, events

    def evaluate_job(self, job_id: str) -> MMOUEvaluationSummary | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None

            rows = _read_jsonl(Path(record.job_dir) / "records" / "predictions.jsonl")
            ready_rows = [row for row in rows if _is_completed_prediction(row)]
            if not ready_rows:
                raise ValueError("No answered MMOU predictions are available to score yet.")

            api = load_mmou_benchmark_api(record.request.benchmarks_dir)
            job_dir = Path(record.job_dir)
            records_dir = job_dir / "records"
            submissions_dir = job_dir / "submissions"
            judge_dir = job_dir / "judge"
            records_dir.mkdir(parents=True, exist_ok=True)
            submissions_dir.mkdir(parents=True, exist_ok=True)
            judge_dir.mkdir(parents=True, exist_ok=True)

            snapshot_path = records_dir / "predictions_answered_current.jsonl"
            _write_jsonl(snapshot_path, ready_rows)
            exported = api.export_submission_from_predictions(
                snapshot_path,
                submissions_dir,
                stem=f"{_safe_name(record.run_name)}-answered-current",
            )
            submission_file = Path(exported["json_path"])
            submission_rows = int(exported.get("rows") or len(ready_rows))

        evaluation = api.evaluate_submission_with_api(
            submission_file=submission_file,
            output_dir=judge_dir,
            evaluator_space=api.MMOU_EVAL_SPACE,
            hf_token=os.getenv("HF_TOKEN") or None,
        )
        summary = _compact_evaluation_summary(evaluation, submission_rows=submission_rows)

        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                summary_path = Path(current.job_dir) / "judge" / "mmou_eval_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
                self._touch(
                    current,
                    stdout=f"Scored {summary.answered} answered MMOU question(s): {summary.answered_accuracy_pct:.1f}%.",
                )
                self._persist_record(current)
        return summary

    def evaluate_question(self, job_id: str, question_id: str) -> MMOUQuestionEvaluationSummary | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if question_id not in record.questions:
                return None

            rows = _read_jsonl(Path(record.job_dir) / "records" / "predictions.jsonl")
            ready_rows = [
                row
                for row in rows
                if str(row.get("question_id") or "") == question_id and _is_completed_prediction(row)
            ]
            if not ready_rows:
                raise ValueError(f"No answered MMOU prediction is available for question {question_id}.")

            row = ready_rows[-1]
            answer = str(row["answer"]).strip()
            api = load_mmou_benchmark_api(record.request.benchmarks_dir)
            job_dir = Path(record.job_dir)
            records_dir = job_dir / "records"
            submissions_dir = job_dir / "submissions"
            judge_dir = job_dir / "judge"
            records_dir.mkdir(parents=True, exist_ok=True)
            submissions_dir.mkdir(parents=True, exist_ok=True)
            judge_dir.mkdir(parents=True, exist_ok=True)

            safe_question_id = _safe_name(question_id)
            snapshot_path = records_dir / f"prediction_{safe_question_id}_current.jsonl"
            _write_jsonl(snapshot_path, [row])
            exported = api.export_submission_from_predictions(
                snapshot_path,
                submissions_dir,
                stem=f"{_safe_name(record.run_name)}-{safe_question_id}-current",
            )
            submission_file = Path(exported["json_path"])
            submission_rows = int(exported.get("rows") or 1)

        evaluation = api.evaluate_submission_with_api(
            submission_file=submission_file,
            output_dir=judge_dir,
            evaluator_space=api.MMOU_EVAL_SPACE,
            hf_token=os.getenv("HF_TOKEN") or None,
        )
        summary = _compact_question_evaluation_summary(
            question_id,
            answer,
            evaluation,
            submission_rows=submission_rows,
        )

        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                summary_path = Path(current.job_dir) / "judge" / "questions" / f"{_safe_name(question_id)}.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
                verdict = "correct" if summary.correct else "incorrect"
                self._touch(current, stdout=f"Scored MMOU question {question_id}: {verdict}.")
                self._persist_record(current)
        return summary

    def _run_job(self, record: MMOUJobRecord) -> None:
        with self._lock:
            record.status = "stopping" if self._stop_requested(record) else "running"
            record.started_at = record.started_at or time.time()
            self._touch(record, stdout=f"Starting MMOU job {record.job_id}.")
            self._persist_record(record)

        try:
            completed = self._completed_question_ids(record)
            with self._lock:
                for question_id in completed:
                    if question_id in record.questions:
                        record.questions[question_id].status = "complete"
                self._persist_record(record)

            pending_questions = deque(
                record.questions[question_id]
                for question_id in record.question_ids
                if question_id not in completed and record.questions[question_id].status != "complete"
            )
            if not pending_questions:
                with self._lock:
                    record.status = "complete"
                    record.finished_at = time.time()
                    self._touch(record, stdout="MMOU job already has complete predictions for every selected question.")
                    self._write_summary(record)
                    self._persist_record(record)
                return

            with ThreadPoolExecutor(max_workers=record.request.workers) as pool:
                in_flight: dict[Future[dict[str, Any]], MMOUQuestionRecord] = {}
                while pending_questions or in_flight:
                    with self._lock:
                        if self._stop_requested(record) and record.status != "stopping":
                            record.status = "stopping"
                            self._touch(record, stdout=record.stop_reason or "Stop requested.")
                            self._persist_record(record)

                    while pending_questions and len(in_flight) < record.request.workers and not self._stop_requested(record):
                        question = pending_questions.popleft()
                        future = pool.submit(self._run_single_question, record, question)
                        in_flight[future] = question

                    if not in_flight:
                        break

                    done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        question = in_flight.pop(future)
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = self._mark_question_error(
                                record,
                                question,
                                error=f"Question worker error: {exc}",
                                attempts=1,
                            )
                        question.prediction_row = row
                        question.result_path = str(Path(record.job_dir) / "records" / "predictions.jsonl")
                        with self._lock:
                            self._persist_predictions(record)
                            self._write_summary(record)
                            self._persist_record(record)
                            self._touch(record, stdout=f"Finished MMOU question {question.question_id}.")

                if self._stop_requested(record):
                    stopped_at = time.time()
                    with self._lock:
                        while pending_questions:
                            question = pending_questions.popleft()
                            question.status = "stopped"
                            question.finished_at = stopped_at
                        for question in record.questions.values():
                            if question.status == "pending":
                                question.status = "stopped"
                                question.finished_at = stopped_at
                        self._persist_record(record)

            with self._lock:
                if self._stop_requested(record):
                    record.status = "stopped"
                else:
                    completed_count = sum(1 for question in record.questions.values() if question.status == "complete")
                    error_count = sum(1 for question in record.questions.values() if question.status == "error")
                    record.status = "error" if error_count and completed_count == 0 else "complete"
                record.finished_at = time.time()
                summary = self._summary_payload(record)
                self._touch(
                    record,
                    stdout=(
                        f"Finished {record.run_name}: {summary['completed_questions']} complete, "
                        f"{summary['error_questions']} error(s)."
                    ),
                )
                self._write_summary(record)
                self._persist_record(record)
        except Exception as exc:
            with self._lock:
                record.status = "error"
                record.finished_at = time.time()
                self._touch(record, stderr=str(exc))
                self._write_summary(record)
                self._persist_record(record)

    def _run_single_question(self, record: MMOUJobRecord, question: MMOUQuestionRecord) -> dict[str, Any]:
        api = load_mmou_benchmark_api(record.request.benchmarks_dir)
        sample = api.MMOUSample(
            question_id=question.question_id,
            question=question.question,
            options=dict(question.options),
            video_url=question.video_url,
            domain=question.domain,
            subdomain=question.subdomain,
            question_type=list(question.question_type),
            start_time=question.start_time,
            end_time=question.end_time,
            video_duration=question.video_duration,
        )
        run_id = f"{record.job_id}_{_safe_name(question.question_id)}_{uuid4().hex[:8]}"
        trace_path = Path(record.job_dir) / "traces" / f"{_safe_name(question.question_id)}.json"

        with self._lock:
            question.status = "running"
            question.started_at = time.time()
            question.finished_at = None
            question.run_id = run_id
            question.trace_path = str(trace_path)
            question.error = None
            question.parse_error = None
            question.tracer = None
            self._touch(record, stdout=f"Running MMOU question {question.question_id} ({question.domain}).")
            self._persist_record(record)

        start = time.time()

        def attach_agent(agent: Any, _metadata: dict[str, Any]) -> None:
            question.tracer = getattr(agent, "_tracer", None)
            if isinstance(_metadata.get("run_id"), str):
                question.run_id = _metadata["run_id"]

        try:
            with tempfile.TemporaryDirectory(prefix=f"mmou-video-{_safe_name(question.question_id)[:16]}-") as video_dir:
                source_video_path = api.download_remote_video(sample, Path(video_dir))
                artifact_root = Path(record.job_dir) / "artifacts" / _safe_name(question.question_id)
                adapter = SanjayaMMOUAdapter(
                    name=MODEL_NAME,
                    root_model=DEFAULT_ROOT_MODEL,
                    sub_model=DEFAULT_SUB_MODEL,
                    recursive_model=DEFAULT_ROOT_MODEL,
                    vision_model=DEFAULT_VISION_MODEL,
                    audio_model=DEFAULT_AUDIO_MODEL,
                    max_iterations=record.request.max_iterations,
                    max_depth=record.request.max_depth,
                    max_budget_usd=record.request.max_budget_usd,
                    max_timeout_s=record.request.max_timeout_s,
                    tracing=True,
                    keep_artifacts=record.request.keep_artifacts,
                    artifacts_root=str(artifact_root),
                    agent_callback=attach_agent,
                )
                response = adapter.generate(
                    api.GenerationRequest(
                        model=MODEL_NAME,
                        system_prompt="You are a careful multimodal evaluator.",
                        prompt=api.build_mmou_prompt(sample),
                        media=[api.MediaInput(type=api.MediaType.video, path=source_video_path)],
                        metadata={"benchmark": "mmou", "question_id": question.question_id},
                    )
                )

            raw_text = str(getattr(response, "text", "") or "")
            try:
                answer = api.parse_answer_letter(raw_text)
                parse_error = None
            except ValueError as exc:
                answer = None
                parse_error = str(exc)

            usage = dict(getattr(response, "usage", {}) or {})
            elapsed = time.time() - start
            row = _prediction_row(
                question,
                video_ref=question.video_url,
                answer=answer,
                raw_text=raw_text,
                cache_key=getattr(response, "cache_key", None),
                cached=bool(getattr(response, "cached", False)),
                usage=usage,
                parse_error=parse_error,
                attempts=1,
                error=parse_error,
            )
            with self._lock:
                question.answer = answer
                question.raw_text = raw_text
                question.parse_error = parse_error
                question.attempts = 1
                question.iterations = int(usage["iterations"]) if usage.get("iterations") is not None else None
                question.cost_usd = float(usage["cost_usd"]) if usage.get("cost_usd") is not None else None
                question.wall_time_s = float(usage.get("wall_time_s", elapsed) or elapsed)
                question.error = parse_error
                question.status = "complete" if answer else "error"
                question.finished_at = time.time()
                self._persist_question_trace(question)
            return row
        except Exception as exc:
            return self._mark_question_error(record, question, error=str(exc), attempts=1)

    def _mark_question_error(
        self,
        record: MMOUJobRecord,
        question: MMOUQuestionRecord,
        *,
        error: str,
        attempts: int,
    ) -> dict[str, Any]:
        row = _prediction_row(
            question,
            video_ref=question.video_url,
            answer=None,
            raw_text=None,
            cache_key=None,
            cached=False,
            usage={},
            parse_error=None,
            attempts=attempts,
            error=error,
        )
        with self._lock:
            question.status = "error"
            question.finished_at = time.time()
            question.error = error
            question.attempts = attempts
            question.prediction_row = row
            self._persist_question_trace(question)
            self._touch(record, stderr=f"MMOU question {question.question_id} failed: {error}")
        return row

    def _completed_question_ids(self, record: MMOUJobRecord) -> set[str]:
        rows = _read_jsonl(Path(record.job_dir) / "records" / "predictions.jsonl")
        completed = {str(row["question_id"]) for row in rows if _is_completed_prediction(row)}
        for row in rows:
            question_id = str(row.get("question_id", ""))
            if question_id in record.questions:
                record.questions[question_id].prediction_row = row
                if _is_completed_prediction(row):
                    record.questions[question_id].answer = str(row["answer"]).strip().upper()
                    record.questions[question_id].status = "complete"
        return completed

    def _stop_requested(self, record: MMOUJobRecord) -> bool:
        return record.stop_requested_at is not None

    def _touch(
        self,
        record: MMOUJobRecord,
        *,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        record.revision += 1
        if stdout:
            record.stdout_tail.append(stdout)
        if stderr:
            record.stderr_tail.append(stderr)

    def _serialize_question(self, record: MMOUJobRecord, question: MMOUQuestionRecord) -> MMOUQuestionStatus:
        trace_count = len(question.trace_events)
        if question.tracer is not None:
            trace_count = len(question.tracer.events)
        return MMOUQuestionStatus(
            question_id=question.question_id,
            question=question.question,
            options=dict(question.options),
            domain=question.domain,
            subdomain=question.subdomain,
            question_type=list(question.question_type),
            start_time=question.start_time,
            end_time=question.end_time,
            status=question.status,  # type: ignore[arg-type]
            started_at=question.started_at,
            finished_at=question.finished_at,
            run_id=question.run_id,
            result_path=question.result_path,
            trace_path=question.trace_path,
            trace_event_count=trace_count,
            answer=question.answer,
            raw_text=question.raw_text,
            parse_error=question.parse_error,
            attempts=question.attempts,
            iterations=question.iterations,
            cost_usd=question.cost_usd,
            wall_time_s=question.wall_time_s,
            error=question.error,
            latest_evaluation=self._load_latest_question_evaluation(record, question),
        )

    def _serialize_job(self, record: MMOUJobRecord) -> MMOUJobSummary:
        questions = [self._serialize_question(record, record.questions[question_id]) for question_id in record.question_ids]
        completed = sum(1 for question in questions if question.status == "complete")
        errors = sum(1 for question in questions if question.status == "error")
        active = [question.question_id for question in questions if question.status == "running"]
        return MMOUJobSummary(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            stop_requested_at=record.stop_requested_at,
            stop_reason=record.stop_reason,
            run_name=record.run_name,
            output_dir=record.output_dir,
            job_dir=record.job_dir,
            models=dict(record.models),
            workers=record.request.workers,
            max_iterations=record.request.max_iterations,
            max_depth=record.request.max_depth,
            max_budget_usd=record.request.max_budget_usd,
            max_timeout_s=record.request.max_timeout_s,
            limit=record.request.limit,
            stratified=record.request.stratified,
            domains=record.request.domains,
            selection_source="question_ids" if record.selection_source == "question_ids" else "dataset",
            keep_artifacts=record.request.keep_artifacts,
            total_questions=len(record.question_ids),
            completed_questions=completed,
            error_questions=errors,
            active_question_ids=active,
            question_ids=list(record.question_ids),
            questions=questions,
            stdout_tail=list(record.stdout_tail),
            stderr_tail=list(record.stderr_tail),
            revision=record.revision,
            latest_evaluation=self._load_latest_evaluation(record),
        )

    def _load_latest_evaluation(self, record: MMOUJobRecord) -> MMOUEvaluationSummary | None:
        payload = _read_json(Path(record.job_dir) / "judge" / "mmou_eval_summary.json", None)
        if not isinstance(payload, dict):
            return None
        try:
            return MMOUEvaluationSummary(**payload)
        except Exception:
            return None

    def _load_latest_question_evaluation(
        self,
        record: MMOUJobRecord,
        question: MMOUQuestionRecord,
    ) -> MMOUQuestionEvaluationSummary | None:
        payload = _read_json(
            Path(record.job_dir) / "judge" / "questions" / f"{_safe_name(question.question_id)}.json",
            None,
        )
        if not isinstance(payload, dict):
            return None
        try:
            return MMOUQuestionEvaluationSummary(**payload)
        except Exception:
            return None

    def _ensure_layout(self, job_dir: Path) -> None:
        for child in (job_dir, job_dir / "records", job_dir / "traces", job_dir / "submissions", job_dir / "judge"):
            child.mkdir(parents=True, exist_ok=True)

    def _resolve_job_output_dir(self, output_dir: str | None) -> Path:
        if not output_dir:
            return self._output_root
        return _resolve_output_dir(output_dir)

    def _job_path(self, record: MMOUJobRecord) -> Path:
        return Path(record.job_dir)

    def _register_job_dir(self, job_dir: Path) -> None:
        index_path = self._output_root / "index.json"
        self._output_root.mkdir(parents=True, exist_ok=True)
        existing = _read_json(index_path, [])
        paths = [str(Path(path)) for path in existing] if isinstance(existing, list) else []
        path_str = str(job_dir)
        if path_str not in paths:
            paths.append(path_str)
        index_path.write_text(json.dumps(paths, indent=2), encoding="utf-8")

    def _candidate_job_dirs(self) -> list[Path]:
        candidates: set[Path] = set()
        if self._output_root.exists():
            candidates.update(path for path in self._output_root.iterdir() if path.is_dir())
        index_path = self._output_root / "index.json"
        for raw_path in _read_json(index_path, []):
            if isinstance(raw_path, str):
                candidates.add(Path(raw_path))
        return sorted(candidates)

    def _hydrate_persisted_jobs(self) -> None:
        for job_dir in self._candidate_job_dirs():
            record = self._load_record(job_dir)
            if record is None:
                continue
            if record.status in ("pending", "running", "stopping"):
                record.status = "interrupted"
                record.finished_at = record.finished_at or time.time()
                for question in record.questions.values():
                    if question.status == "running":
                        question.status = "error"
                        question.error = question.error or "Interrupted before completion."
                self._touch(record, stderr="Job was interrupted while the API was offline.")
                self._persist_record(record)
            self._jobs[record.job_id] = record

    def _load_record(self, job_dir: Path) -> MMOUJobRecord | None:
        job_payload = _read_json(job_dir / "job.json", None)
        questions_payload = _read_json(job_dir / "questions.json", None)
        if not isinstance(job_payload, dict) or not isinstance(questions_payload, list):
            return None

        try:
            request = MMOUJobCreateRequest(**job_payload["request"])
            question_records = {}
            for item in questions_payload:
                if not isinstance(item, dict):
                    continue
                trace_events = item.pop("trace_events", [])
                prediction_row = item.pop("prediction_row", None)
                item.pop("tracer", None)
                question = MMOUQuestionRecord(**item)
                question.trace_events = trace_events if isinstance(trace_events, list) else []
                question.prediction_row = prediction_row if isinstance(prediction_row, dict) else None
                question_records[question.question_id] = question

            record = MMOUJobRecord(
                job_id=str(job_payload["job_id"]),
                request=request,
                run_name=str(job_payload["run_name"]),
                output_dir=str(job_payload["output_dir"]),
                job_dir=str(job_dir),
                models=dict(job_payload.get("models") or {}),
                question_ids=list(job_payload["question_ids"]),
                questions=question_records,
                selection_source=str(job_payload.get("selection_source") or "dataset"),
                status=str(job_payload.get("status") or "interrupted"),
                created_at=float(job_payload.get("created_at") or time.time()),
                started_at=job_payload.get("started_at"),
                finished_at=job_payload.get("finished_at"),
                stop_requested_at=job_payload.get("stop_requested_at"),
                stop_reason=job_payload.get("stop_reason"),
                stdout_tail=deque(job_payload.get("stdout_tail") or [], maxlen=200),
                stderr_tail=deque(job_payload.get("stderr_tail") or [], maxlen=200),
                revision=int(job_payload.get("revision") or 0),
            )
        except Exception:
            return None

        self._completed_question_ids(record)
        return record

    def _persist_record(self, record: MMOUJobRecord) -> None:
        job_dir = self._job_path(record)
        self._ensure_layout(job_dir)
        job_payload = {
            "job_id": record.job_id,
            "request": record.request.model_dump(),
            "run_name": record.run_name,
            "output_dir": record.output_dir,
            "job_dir": record.job_dir,
            "models": record.models,
            "question_ids": record.question_ids,
            "selection_source": record.selection_source,
            "status": record.status,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "stop_requested_at": record.stop_requested_at,
            "stop_reason": record.stop_reason,
            "stdout_tail": list(record.stdout_tail),
            "stderr_tail": list(record.stderr_tail),
            "revision": record.revision,
        }
        (job_dir / "job.json").write_text(json.dumps(job_payload, indent=2, default=str), encoding="utf-8")
        questions_payload = [self._question_payload(record.questions[question_id]) for question_id in record.question_ids]
        (job_dir / "questions.json").write_text(json.dumps(questions_payload, indent=2, default=str), encoding="utf-8")

    def _question_payload(self, question: MMOUQuestionRecord) -> dict[str, Any]:
        return {
            "question_id": question.question_id,
            "question": question.question,
            "options": dict(question.options),
            "video_url": question.video_url,
            "domain": question.domain,
            "subdomain": question.subdomain,
            "question_type": list(question.question_type),
            "start_time": question.start_time,
            "end_time": question.end_time,
            "video_duration": question.video_duration,
            "status": question.status,
            "started_at": question.started_at,
            "finished_at": question.finished_at,
            "run_id": question.run_id,
            "result_path": question.result_path,
            "trace_path": question.trace_path,
            "answer": question.answer,
            "raw_text": question.raw_text,
            "parse_error": question.parse_error,
            "attempts": question.attempts,
            "iterations": question.iterations,
            "cost_usd": question.cost_usd,
            "wall_time_s": question.wall_time_s,
            "error": question.error,
            "trace_events": list(question.trace_events),
            "prediction_row": question.prediction_row,
        }

    def _persist_predictions(self, record: MMOUJobRecord) -> None:
        rows = [
            record.questions[question_id].prediction_row
            for question_id in record.question_ids
            if record.questions[question_id].prediction_row is not None
        ]
        _write_jsonl(Path(record.job_dir) / "records" / "predictions.jsonl", rows)

    def _normalize_event(self, raw: dict[str, Any]) -> dict[str, Any]:
        kind, timestamp, payload = normalize_trace_event(raw)
        return {"kind": kind, "timestamp": timestamp, "payload": payload}

    def _persist_question_trace(self, question: MMOUQuestionRecord) -> None:
        if question.tracer is not None:
            dump_events = getattr(question.tracer, "dump_events", None)
            raw_events = dump_events() if callable(dump_events) else list(question.tracer.events)
            question.trace_events = [self._normalize_event(raw) for raw in raw_events]
        if question.trace_path:
            path = Path(question.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "question_id": question.question_id,
                        "run_id": question.run_id,
                        "events": question.trace_events,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    def _summary_payload(self, record: MMOUJobRecord) -> dict[str, Any]:
        completed = [question for question in record.questions.values() if question.status == "complete"]
        errors = [question for question in record.questions.values() if question.status == "error"]
        return {
            "job_id": record.job_id,
            "run_name": record.run_name,
            "status": record.status,
            "total_questions": len(record.question_ids),
            "completed_questions": len(completed),
            "error_questions": len(errors),
            "selected_domains": sorted({record.questions[question_id].domain for question_id in record.question_ids}),
            "domain_counts": self._domain_counts(record),
            "total_cost_usd": sum(question.cost_usd or 0.0 for question in completed),
            "total_wall_time_s": sum(question.wall_time_s or 0.0 for question in completed),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _write_summary(self, record: MMOUJobRecord) -> None:
        summary = self._summary_payload(record)
        (Path(record.job_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _domain_counts(self, record: MMOUJobRecord) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question_id in record.question_ids:
            domain = record.questions[question_id].domain
            counts[domain] = counts.get(domain, 0) + 1
        return dict(sorted(counts.items()))
