"""Background service for video benchmark batch jobs."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from sanjaya import Agent
from sanjaya.tools.video import VideoToolkit
from sanjaya.tools.video.media import MediaToolError, video_duration_seconds

from sanjaya_api.models import (
    BenchmarkCatalogResponse,
    BenchmarkJobCreateRequest,
    BenchmarkJobSummary,
    BenchmarkPromptCatalogItem,
    BenchmarkPromptStatus,
)
from sanjaya_api.trace_events import normalize_trace_event

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "sanjaya_artifacts"


@lru_cache(maxsize=1)
def load_video_benchmark_module() -> ModuleType:
    """Load the existing video benchmark runner as a module."""
    module_path = PROJECT_ROOT / "scripts" / "run_video_benchmarks.py"
    spec = importlib.util.spec_from_file_location("sanjaya_video_benchmarks", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompt_group(prompt: dict[str, Any]) -> str:
    return "lvb" if prompt.get("is_mcq") else "demo"


def _resolve_effective_limits(request: BenchmarkJobCreateRequest) -> tuple[int, float]:
    if request.fast:
        max_iterations = request.max_iterations if request.max_iterations != 20 else 10
        max_budget = request.max_budget_usd if request.max_budget_usd != 1.0 else 0.50
        return max_iterations, max_budget
    return request.max_iterations, request.max_budget_usd


def _resolve_output_dir(output_dir: str | None, module: ModuleType) -> Path:
    if not output_dir:
        return Path(module.RESULTS_DIR)
    candidate = Path(output_dir)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _is_valid_video_path(video_path: Path) -> tuple[bool, str | None]:
    """Return whether a video is readable by ffprobe."""
    if not video_path.exists():
        return False, f"Missing video: {video_path}"
    try:
        duration_s = video_duration_seconds(str(video_path))
    except (MediaToolError, FileNotFoundError) as exc:
        return False, str(exc)
    if duration_s <= 0:
        return False, f"Invalid video duration: {duration_s}"
    return True, None


@dataclass
class BenchmarkPromptRecord:
    prompt_id: int
    prompt_name: str
    video_key: str
    question: str
    is_mcq: bool
    group: str
    status: str = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    run_id: str | None = None
    result_path: str | None = None
    trace_path: str | None = None
    iterations: int | None = None
    cost_usd: float | None = None
    wall_time_s: float | None = None
    error: str | None = None
    mcq_correct: bool | None = None
    tracer: Any | None = None
    raw_trace_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkJobRecord:
    job_id: str
    request: BenchmarkJobCreateRequest
    run_name: str
    output_dir: str
    models: dict[str, str | None]
    effective_max_iterations: int
    effective_max_budget_usd: float
    prompt_ids: list[int]
    prompts: dict[int, BenchmarkPromptRecord]
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


class BenchmarkJobService:
    """Manages benchmark batch jobs and prompt-level live traces."""

    def __init__(self) -> None:
        self._jobs: dict[str, BenchmarkJobRecord] = {}
        self._lock = threading.RLock()

    def get_catalog(self) -> BenchmarkCatalogResponse:
        module = load_video_benchmark_module()
        prompts = [
            BenchmarkPromptCatalogItem(
                prompt_id=prompt["id"],
                prompt_name=prompt["name"],
                video_key=prompt["video_key"],
                question=module._format_mcq_question(
                    module.LVB_QUESTIONS[prompt["id"]]["question"],
                    module.LVB_QUESTIONS[prompt["id"]]["candidates"],
                ) if prompt.get("is_mcq") else prompt["question"],
                is_mcq=bool(prompt.get("is_mcq", False)),
                group=_prompt_group(prompt),
            )
            for prompt in module.PROMPTS
        ]
        defaults = {
            "workers": 6,
            "max_iterations": 20,
            "max_budget_usd": 1.0,
            "fast": False,
            "download_lvb": False,
            "output_dir": str(Path(module.RESULTS_DIR)),
            "models": {
                "root": module.ROOT_MODEL,
                "sub": module.SUB_MODEL,
                "vision": module.VISION_MODEL,
                "caption": module.CAPTION_MODEL,
                "critic": module.CRITIC_MODEL,
            },
            "prompt_presets": {
                "all": [prompt["id"] for prompt in module.PROMPTS],
                "demo": [prompt["id"] for prompt in module.PROMPTS if not prompt.get("is_mcq")],
                "lvb": [prompt["id"] for prompt in module.PROMPTS if prompt.get("is_mcq")],
            },
        }
        return BenchmarkCatalogResponse(prompts=prompts, defaults=defaults)

    def start_job(self, request: BenchmarkJobCreateRequest) -> BenchmarkJobSummary:
        module = load_video_benchmark_module()
        prompt_lookup = {prompt["id"]: prompt for prompt in module.PROMPTS}

        prompt_ids = request.prompt_ids or [prompt["id"] for prompt in module.PROMPTS]
        invalid_ids = sorted(set(prompt_ids) - set(prompt_lookup))
        if invalid_ids:
            raise ValueError(f"Unknown prompt ids: {invalid_ids}")

        run_name = request.run_name or f"video_bench_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = str(_resolve_output_dir(request.output_dir, module))
        max_iterations, max_budget = _resolve_effective_limits(request)
        job_id = f"benchmark_job_{uuid4().hex[:12]}"
        models = {
            "root": module.ROOT_MODEL,
            "sub": module.SUB_MODEL,
            "vision": module.VISION_MODEL,
            "caption": module.CAPTION_MODEL,
            "critic": module.CRITIC_MODEL,
        }

        prompt_records = {
            prompt_id: BenchmarkPromptRecord(
                prompt_id=prompt_id,
                prompt_name=prompt_lookup[prompt_id]["name"],
                video_key=prompt_lookup[prompt_id]["video_key"],
                question=prompt_lookup[prompt_id]["question"],
                is_mcq=bool(prompt_lookup[prompt_id].get("is_mcq", False)),
                group=_prompt_group(prompt_lookup[prompt_id]),
            )
            for prompt_id in prompt_ids
        }
        record = BenchmarkJobRecord(
            job_id=job_id,
            request=request,
            run_name=run_name,
            output_dir=output_dir,
            models=models,
            effective_max_iterations=max_iterations,
            effective_max_budget_usd=max_budget,
            prompt_ids=prompt_ids,
            prompts=prompt_records,
        )

        with self._lock:
            self._jobs[job_id] = record
            self._touch(record, stdout=f"Queued {len(prompt_ids)} prompt(s) for run {run_name}.")

        thread = threading.Thread(target=self._run_job, args=(record,), daemon=True)
        record.thread = thread
        thread.start()
        return self._serialize_job(record)

    def list_jobs(self) -> list[BenchmarkJobSummary]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._serialize_job(job) for job in jobs]

    def get_job(self, job_id: str) -> BenchmarkJobSummary | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return self._serialize_job(record) if record else None

    def get_job_record(self, job_id: str) -> BenchmarkJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def request_stop(self, job_id: str, *, reason: str = "Stop requested from dashboard") -> BenchmarkJobSummary | None:
        """Request cooperative stop for a benchmark job."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None

            if record.status in ("complete", "error", "stopped"):
                return self._serialize_job(record)

            if record.stop_requested_at is None:
                record.stop_requested_at = time.time()
                record.stop_reason = reason
                record.status = "stopping"
                self._touch(record, stdout=reason)

            return self._serialize_job(record)

    def get_prompt_trace(self, job_id: str, prompt_id: int) -> tuple[str | None, list[dict[str, Any]]] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            prompt = record.prompts.get(prompt_id)
            if prompt is None:
                return None
            raw_events = list(prompt.raw_trace_events)
            if prompt.tracer is not None:
                raw_events = list(prompt.tracer.events)
            normalized = []
            for raw in raw_events:
                kind, timestamp, payload = normalize_trace_event(raw)
                normalized.append({"kind": kind, "timestamp": timestamp, "payload": payload})
            return prompt.run_id, normalized

    def _touch(
        self,
        record: BenchmarkJobRecord,
        *,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        record.revision += 1
        if stdout:
            record.stdout_tail.append(stdout)
        if stderr:
            record.stderr_tail.append(stderr)

    @staticmethod
    def _stop_requested(record: BenchmarkJobRecord) -> bool:
        return record.stop_requested_at is not None

    def _serialize_prompt(self, prompt: BenchmarkPromptRecord) -> BenchmarkPromptStatus:
        trace_event_count = len(prompt.raw_trace_events)
        if prompt.tracer is not None:
            trace_event_count = len(prompt.tracer.events)

        return BenchmarkPromptStatus(
            prompt_id=prompt.prompt_id,
            prompt_name=prompt.prompt_name,
            video_key=prompt.video_key,
            question=prompt.question,
            is_mcq=prompt.is_mcq,
            group="lvb" if prompt.group == "lvb" else "demo",
            status=prompt.status,
            started_at=prompt.started_at,
            finished_at=prompt.finished_at,
            run_id=prompt.run_id,
            result_path=prompt.result_path,
            trace_path=prompt.trace_path,
            trace_event_count=trace_event_count,
            iterations=prompt.iterations,
            cost_usd=prompt.cost_usd,
            wall_time_s=prompt.wall_time_s,
            error=prompt.error,
            mcq_correct=prompt.mcq_correct,
        )

    def _serialize_job(self, record: BenchmarkJobRecord | None) -> BenchmarkJobSummary:
        if record is None:
            raise ValueError("record must not be None")

        prompts = [self._serialize_prompt(record.prompts[prompt_id]) for prompt_id in record.prompt_ids]
        completed_prompts = sum(1 for prompt in prompts if prompt.status == "complete")
        error_prompts = sum(1 for prompt in prompts if prompt.status == "error")
        active_prompt_ids = [prompt.prompt_id for prompt in prompts if prompt.status == "running"]

        return BenchmarkJobSummary(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            stop_requested_at=record.stop_requested_at,
            stop_reason=record.stop_reason,
            run_name=record.run_name,
            output_dir=record.output_dir,
            models=dict(record.models),
            workers=record.request.workers,
            max_iterations=record.effective_max_iterations,
            max_budget_usd=record.effective_max_budget_usd,
            fast=record.request.fast,
            download_lvb=record.request.download_lvb,
            total_prompts=len(record.prompt_ids),
            completed_prompts=completed_prompts,
            error_prompts=error_prompts,
            active_prompt_ids=active_prompt_ids,
            prompt_ids=list(record.prompt_ids),
            prompts=prompts,
            stdout_tail=list(record.stdout_tail),
            stderr_tail=list(record.stderr_tail),
            revision=record.revision,
        )

    def _run_job(self, record: BenchmarkJobRecord) -> None:
        module = load_video_benchmark_module()
        prompt_lookup = {prompt["id"]: prompt for prompt in module.PROMPTS}
        prompts_to_run = [prompt_lookup[prompt_id] for prompt_id in record.prompt_ids]

        with self._lock:
            record.status = "stopping" if self._stop_requested(record) else "running"
            record.started_at = time.time()
            self._touch(record, stdout=f"Starting benchmark job {record.job_id}.")

        try:
            if record.request.download_lvb:
                invalid_lvb_paths: list[Path] = []
                seen_lvb_paths: set[Path] = set()
                for prompt in prompts_to_run:
                    if not prompt.get("is_mcq"):
                        continue
                    video_path = Path(module.VIDEOS[prompt["video_key"]]["video"])
                    if video_path in seen_lvb_paths:
                        continue
                    seen_lvb_paths.add(video_path)
                    is_valid, _error = _is_valid_video_path(video_path)
                    if not is_valid and video_path.exists():
                        invalid_lvb_paths.append(video_path)

                for invalid_path in invalid_lvb_paths:
                    try:
                        invalid_path.unlink()
                        with self._lock:
                            self._touch(record, stdout=f"Removed invalid cached video before re-download: {invalid_path.name}")
                    except OSError as exc:
                        with self._lock:
                            self._touch(record, stderr=f"Could not remove invalid cached video {invalid_path}: {exc}")

                with self._lock:
                    self._touch(record, stdout="Downloading missing LongVideoBench videos before execution.")
                module.download_lvb_videos()

            available_prompts: list[dict[str, Any]] = []
            invalid_results: list[dict[str, Any]] = []
            for prompt in prompts_to_run:
                video_path = Path(module.VIDEOS[prompt["video_key"]]["video"])
                is_valid, error = _is_valid_video_path(video_path)
                if is_valid:
                    available_prompts.append(prompt)
                    continue

                prompt_record = record.prompts[prompt["id"]]
                prompt_record.status = "error"
                prompt_record.error = error or f"Invalid video: {video_path}"
                prompt_record.finished_at = time.time()
                invalid_results.append({
                    "prompt_id": prompt["id"],
                    "prompt_name": prompt["name"],
                    "video_key": prompt["video_key"],
                    "question": prompt["question"],
                    "is_mcq": bool(prompt.get("is_mcq", False)),
                    "error": prompt_record.error,
                    "cost_usd": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "wall_time_s": 0,
                })
                with self._lock:
                    self._touch(record, stderr=prompt_record.error)

            if not available_prompts and invalid_results:
                raise RuntimeError("No prompts had usable local videos.")

            output_root = Path(record.output_dir)
            run_dir = output_root / record.run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            n_demo = sum(1 for prompt in available_prompts if not prompt.get("is_mcq"))
            n_mcq = sum(1 for prompt in available_prompts if prompt.get("is_mcq"))
            run_config = {
                "run_name": record.run_name,
                "models": {
                    "root": module.ROOT_MODEL,
                    "sub": module.SUB_MODEL,
                    "vision": module.VISION_MODEL,
                    "caption": module.CAPTION_MODEL,
                    "critic": module.CRITIC_MODEL,
                },
                "max_iterations": record.effective_max_iterations,
                "max_budget_usd_per_prompt": record.effective_max_budget_usd,
                "max_depth": 2,
                "workers": record.request.workers,
                "prompts": [prompt["id"] for prompt in available_prompts],
                "n_demo_prompts": n_demo,
                "n_mcq_prompts": n_mcq,
                "fast_mode": record.request.fast,
            }
            (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
            checkpoint_path = run_dir / "checkpoint.jsonl"

            with self._lock:
                self._touch(record, stdout=f"Writing results into {run_dir}.")

            results: list[dict[str, Any]] = list(invalid_results)
            overall_start = time.time()

            with ThreadPoolExecutor(max_workers=record.request.workers) as pool:
                pending_prompts = deque(available_prompts)
                in_flight: dict[Future[dict[str, Any]], dict[str, Any]] = {}
                completed = 0
                while pending_prompts or in_flight:
                    with self._lock:
                        if self._stop_requested(record) and record.status != "stopping":
                            record.status = "stopping"
                            self._touch(record, stdout=record.stop_reason or "Stop requested.")

                    while pending_prompts and len(in_flight) < record.request.workers and not self._stop_requested(record):
                        prompt = pending_prompts.popleft()
                        prompt_record = record.prompts[prompt["id"]]
                        prompt_record.status = "pending"
                        future = pool.submit(
                            self._run_single_prompt,
                            record,
                            prompt_record,
                            prompt,
                        )
                        in_flight[future] = prompt

                    if not in_flight:
                        break

                    done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        prompt = in_flight.pop(future)
                        completed += 1
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "prompt_id": prompt["id"],
                                "prompt_name": prompt["name"],
                                "video_key": prompt["video_key"],
                                "question": prompt["question"],
                                "is_mcq": bool(prompt.get("is_mcq", False)),
                                "error": f"Prompt worker error: {exc}",
                                "cost_usd": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "wall_time_s": 0,
                            }
                            if prompt.get("is_mcq"):
                                result["mcq_correct"] = False

                        results.append(result)
                        out_path = run_dir / f"prompt_{prompt['id']:02d}_{prompt['name']}.json"
                        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                        with checkpoint_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(result, default=str) + "\n")

                        prompt_record = record.prompts[prompt["id"]]
                        prompt_record.result_path = str(out_path)
                        with self._lock:
                            self._touch(
                                record,
                                stdout=f"Completed prompt {prompt['id']} ({completed}/{len(available_prompts)}).",
                            )

                if self._stop_requested(record):
                    stopped_at = time.time()
                    while pending_prompts:
                        prompt = pending_prompts.popleft()
                        prompt_record = record.prompts[prompt["id"]]
                        prompt_record.status = "stopped"
                        prompt_record.finished_at = stopped_at
                    for prompt in record.prompts.values():
                        if prompt.status == "pending":
                            prompt.status = "stopped"
                            prompt.finished_at = stopped_at

            summary = self._write_summary(
                record=record,
                run_dir=run_dir,
                results=results,
                n_demo=n_demo,
                n_mcq=n_mcq,
                overall_start=overall_start,
                run_config=run_config,
            )
            with self._lock:
                if self._stop_requested(record):
                    record.status = "stopped"
                else:
                    record.status = "error" if summary["errors"] > 0 and summary["total_prompts"] == summary["errors"] else "complete"
                record.finished_at = time.time()
                self._touch(
                    record,
                    stdout=(
                        (
                            f"Stopped {record.run_name}: ${summary['total_cost_usd']:.4f}, "
                            f"{summary['total_wall_time_s']:.1f}s, {summary['errors']} error(s)."
                            if self._stop_requested(record)
                            else f"Finished {record.run_name}: ${summary['total_cost_usd']:.4f}, "
                            f"{summary['total_wall_time_s']:.1f}s, {summary['errors']} error(s)."
                        )
                    ),
                )
        except Exception as exc:
            with self._lock:
                record.status = "error"
                record.finished_at = time.time()
                self._touch(record, stderr=str(exc))

    def _run_single_prompt(
        self,
        record: BenchmarkJobRecord,
        prompt_record: BenchmarkPromptRecord,
        prompt: dict[str, Any],
    ) -> dict[str, Any]:
        module = load_video_benchmark_module()
        prompt_id = prompt["id"]
        video_cfg = module.VIDEOS[prompt["video_key"]]
        is_mcq = bool(prompt.get("is_mcq", False))

        if is_mcq:
            mcq_data = module.LVB_QUESTIONS[prompt_id]
            question = module._format_mcq_question(mcq_data["question"], mcq_data["candidates"])
            gt_answer = mcq_data["gt_answer"]
        else:
            question = prompt["question"]
            gt_answer = None

        subtitle_path = video_cfg.get("subtitle")
        if subtitle_path and not Path(subtitle_path).exists():
            subtitle_path = None

        has_existing_sub = subtitle_path is not None or module._check_subtitle_exists(video_cfg["video"])
        run_id = f"{record.job_id}_prompt_{prompt_id:02d}_{uuid4().hex[:8]}"
        trace_path = ARTIFACTS_DIR / run_id / "trace.json"

        with self._lock:
            prompt_record.status = "running"
            prompt_record.started_at = time.time()
            prompt_record.run_id = run_id
            prompt_record.trace_path = str(trace_path)
            self._touch(record, stdout=f"Running prompt {prompt_id}: {prompt_record.prompt_name}")

        start = time.time()
        try:
            agent = Agent(
                model=module.ROOT_MODEL,
                sub_model=module.SUB_MODEL,
                vision_model=module.VISION_MODEL,
                caption_model=module.CAPTION_MODEL,
                critic_model=module.CRITIC_MODEL,
                max_iterations=record.effective_max_iterations,
                max_depth=2,
                max_budget_usd=record.effective_max_budget_usd,
                tracing=True,
            )
            prompt_record.tracer = agent._tracer
            agent.use(VideoToolkit())

            answer = agent.ask(
                question,
                context={
                    "run_id": run_id,
                    "run_type": "benchmark_job",
                    "benchmark_job_id": record.job_id,
                    "benchmark_prompt_id": prompt_id,
                },
                video=video_cfg["video"],
                subtitle=subtitle_path,
            )
            elapsed = time.time() - start

            subtitle_generated = not has_existing_sub and module._check_subtitle_exists(video_cfg["video"])
            subtitle_info = {
                "had_existing_subtitle": has_existing_sub,
                "subtitle_generated": subtitle_generated,
                "subtitle_source": "existing" if has_existing_sub else ("whisper_local" if subtitle_generated else "none"),
            }

            raw_trace_events = list(agent._tracer.dump_events())
            prompt_record.raw_trace_events = raw_trace_events

            mcq_correct = None
            mcq_predicted = None
            if is_mcq and gt_answer:
                raw_text = answer.text or ""
                answer_data = answer.data if isinstance(answer.data, dict) else {}
                mcq_predicted = module._extract_mcq_answer(raw_text, answer_data, mcq_data["candidates"])
                mcq_correct = mcq_predicted is not None and mcq_predicted.strip() == gt_answer.strip()

            result: dict[str, Any] = {
                "prompt_id": prompt_id,
                "prompt_name": prompt["name"],
                "video_key": prompt["video_key"],
                "question": question[:500],
                "is_mcq": is_mcq,
                "trace_run_id": run_id,
                "trace_path": str(trace_path),
                "config": {
                    "root_model": module.ROOT_MODEL,
                    "sub_model": module.SUB_MODEL,
                    "vision_model": module.VISION_MODEL,
                    "caption_model": module.CAPTION_MODEL,
                    "max_depth": 2,
                    "max_budget_usd": record.effective_max_budget_usd,
                    "max_iterations": record.effective_max_iterations,
                },
                "answer_text": answer.text,
                "answer_data": answer.data,
                "iterations": answer.iterations,
                "cost_usd": answer.cost_usd,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "wall_time_s": round(elapsed, 2),
                "evidence_count": len(answer.evidence),
                "evidence_sources": [evidence.source for evidence in answer.evidence],
                "subtitle": subtitle_info,
                "trace_events": raw_trace_events,
            }
            if is_mcq:
                result["mcq_gt_answer"] = gt_answer
                result["mcq_predicted"] = mcq_predicted
                result["mcq_correct"] = mcq_correct

            with self._lock:
                prompt_record.status = "complete"
                prompt_record.finished_at = time.time()
                prompt_record.iterations = answer.iterations
                prompt_record.cost_usd = answer.cost_usd
                prompt_record.wall_time_s = round(elapsed, 2)
                prompt_record.mcq_correct = mcq_correct
                self._touch(
                    record,
                    stdout=(
                        f"Prompt {prompt_id} finished in {elapsed:.1f}s "
                        f"after {answer.iterations} iteration(s)."
                    ),
                )
            return result
        except Exception as exc:
            elapsed = time.time() - start
            raw_trace_events = []
            if prompt_record.tracer is not None:
                raw_trace_events = list(prompt_record.tracer.events)
                prompt_record.raw_trace_events = raw_trace_events

            with self._lock:
                prompt_record.status = "error"
                prompt_record.finished_at = time.time()
                prompt_record.error = str(exc)
                prompt_record.wall_time_s = round(elapsed, 2)
                self._touch(record, stderr=f"Prompt {prompt_id} failed: {exc}")

            result = {
                "prompt_id": prompt_id,
                "prompt_name": prompt["name"],
                "video_key": prompt["video_key"],
                "question": question[:500],
                "is_mcq": is_mcq,
                "trace_run_id": run_id,
                "trace_path": str(trace_path),
                "error": str(exc),
                "cost_usd": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_time_s": round(elapsed, 2),
                "trace_events": raw_trace_events,
            }
            if is_mcq:
                result["mcq_correct"] = False
            return result

    def _write_summary(
        self,
        *,
        record: BenchmarkJobRecord,
        run_dir: Path,
        results: list[dict[str, Any]],
        n_demo: int,
        n_mcq: int,
        overall_start: float,
        run_config: dict[str, Any],
    ) -> dict[str, Any]:
        total_cost = sum(result.get("cost_usd", 0) or 0 for result in results)
        total_time = sum(result.get("wall_time_s", 0) or 0 for result in results)
        total_input = sum(result.get("input_tokens", 0) or 0 for result in results)
        total_output = sum(result.get("output_tokens", 0) or 0 for result in results)
        errors = sum(1 for result in results if result.get("error"))
        avg_iter = sum(result.get("iterations", 0) or 0 for result in results) / max(len(results), 1)

        mcq_results = [result for result in results if result.get("is_mcq")]
        mcq_correct = sum(1 for result in mcq_results if result.get("mcq_correct"))
        mcq_accuracy = mcq_correct / len(mcq_results) if mcq_results else None
        overall_elapsed = time.time() - overall_start

        summary = {
            "run_name": record.run_name,
            "models": run_config["models"],
            "total_prompts": len(results),
            "n_demo": n_demo,
            "n_mcq": n_mcq,
            "mcq_accuracy": round(mcq_accuracy, 4) if mcq_accuracy is not None else None,
            "mcq_correct": mcq_correct,
            "mcq_total": len(mcq_results),
            "errors": errors,
            "total_cost_usd": round(total_cost, 4),
            "total_wall_time_s": round(overall_elapsed, 1),
            "sum_wall_time_s": round(total_time, 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_cost_per_prompt": round(total_cost / max(len(results), 1), 4),
            "avg_iterations": round(avg_iter, 1),
            "per_prompt": [
                {
                    "id": result.get("prompt_id"),
                    "name": result.get("prompt_name"),
                    "video": result.get("video_key"),
                    "is_mcq": result.get("is_mcq", False),
                    "mcq_correct": result.get("mcq_correct"),
                    "cost_usd": result.get("cost_usd", 0),
                    "iterations": result.get("iterations", 0),
                    "wall_time_s": result.get("wall_time_s", 0),
                    "evidence_count": result.get("evidence_count", 0),
                    "error": result.get("error"),
                }
                for result in sorted(results, key=lambda item: item.get("prompt_id", 0))
            ],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary
