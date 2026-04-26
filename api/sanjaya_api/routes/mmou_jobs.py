"""MMOU benchmark job endpoints."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from sanjaya_api.models import (
    MMOUCatalogResponse,
    MMOUEvaluationSummary,
    MMOUJobCreateRequest,
    MMOUJobSummary,
    MMOUQuestionEvaluationSummary,
    MMOUQuestionTraceResponse,
)
from sanjaya_api.services.mmou_jobs import MMOUBenchmarkJobService
from sanjaya_api.sse import format_heartbeat, format_sse_event
from sanjaya_api.trace_events import normalize_trace_event

router = APIRouter()

_mmou_jobs = MMOUBenchmarkJobService()


@router.get("/mmou-jobs/catalog", response_model=MMOUCatalogResponse)
async def get_mmou_catalog() -> MMOUCatalogResponse:
    """Return MMOU metadata and defaults for the launcher UI."""
    try:
        return _mmou_jobs.get_catalog()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/mmou-jobs", response_model=MMOUJobSummary)
async def create_mmou_job(request: MMOUJobCreateRequest) -> MMOUJobSummary:
    """Create a new MMOU benchmark job."""
    try:
        return _mmou_jobs.start_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/mmou-jobs", response_model=list[MMOUJobSummary])
async def list_mmou_jobs() -> list[MMOUJobSummary]:
    """List MMOU jobs newest-first."""
    return _mmou_jobs.list_jobs()


@router.get("/mmou-jobs/{job_id}", response_model=MMOUJobSummary)
async def get_mmou_job(job_id: str) -> MMOUJobSummary:
    """Fetch one MMOU job snapshot."""
    job = _mmou_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"MMOU job {job_id} not found")
    return job


@router.post("/mmou-jobs/{job_id}/stop", response_model=MMOUJobSummary)
async def stop_mmou_job(job_id: str) -> MMOUJobSummary:
    """Request cooperative stop for an MMOU job."""
    job = _mmou_jobs.request_stop(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"MMOU job {job_id} not found")
    return job


@router.post("/mmou-jobs/{job_id}/resume", response_model=MMOUJobSummary)
async def resume_mmou_job(job_id: str) -> MMOUJobSummary:
    """Resume incomplete MMOU questions for a persisted job."""
    job = _mmou_jobs.resume_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"MMOU job {job_id} not found")
    return job


@router.post("/mmou-jobs/{job_id}/evaluate", response_model=MMOUEvaluationSummary)
async def evaluate_mmou_job(job_id: str) -> MMOUEvaluationSummary:
    """Submit currently answered MMOU predictions to the official evaluator."""
    try:
        result = _mmou_jobs.evaluate_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"MMOU job {job_id} not found")
    return result


@router.post(
    "/mmou-jobs/{job_id}/questions/{question_id}/evaluate",
    response_model=MMOUQuestionEvaluationSummary,
)
async def evaluate_mmou_question(job_id: str, question_id: str) -> MMOUQuestionEvaluationSummary:
    """Submit one answered MMOU prediction to the official evaluator."""
    try:
        result = _mmou_jobs.evaluate_question(job_id, question_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"MMOU question {question_id} not found")
    return result


@router.get("/mmou-jobs/{job_id}/questions/{question_id}/trace", response_model=MMOUQuestionTraceResponse)
async def get_mmou_question_trace(job_id: str, question_id: str) -> MMOUQuestionTraceResponse:
    """Fetch the current trace snapshot for one MMOU question."""
    result = _mmou_jobs.get_question_trace(job_id, question_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Trace for MMOU question {question_id} not found")
    run_id, events = result
    return MMOUQuestionTraceResponse(question_id=question_id, run_id=run_id, events=events)


@router.get("/mmou-jobs/{job_id}/events")
async def stream_mmou_job_events(job_id: str) -> EventSourceResponse:
    """Stream MMOU job updates and question trace events."""
    record = _mmou_jobs.get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"MMOU job {job_id} not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        revision = -1
        cursors: dict[str, int] = {}
        heartbeat_interval = 2.0
        poll_interval = 0.25
        time_since_heartbeat = 0.0

        while True:
            current = _mmou_jobs.get_job_record(job_id)
            if current is None:
                yield format_sse_event(
                    kind="stream_error",
                    timestamp=time.time(),
                    payload={"error": f"MMOU job {job_id} disappeared"},
                )
                return

            snapshot = _mmou_jobs.get_job(job_id)
            if snapshot is None:
                yield format_sse_event(
                    kind="stream_error",
                    timestamp=time.time(),
                    payload={"error": f"MMOU job {job_id} disappeared"},
                )
                return

            if snapshot.revision != revision:
                revision = snapshot.revision
                yield format_sse_event(
                    kind="mmou_job_update",
                    timestamp=time.time(),
                    payload=snapshot.model_dump(),
                )

            for question_id in current.question_ids:
                question = current.questions[question_id]
                tracer = question.tracer
                if tracer is None:
                    continue
                cursor = cursors.get(question_id, 0)
                events = tracer.events
                while cursor < len(events):
                    raw = events[cursor]
                    cursor += 1
                    kind, timestamp, payload = normalize_trace_event(raw)
                    yield format_sse_event(
                        kind="mmou_trace_event",
                        timestamp=timestamp,
                        payload={
                            "job_id": job_id,
                            "question_id": question_id,
                            "event": {
                                "kind": kind,
                                "timestamp": timestamp,
                                "payload": payload,
                            },
                        },
                    )
                cursors[question_id] = cursor

            if snapshot.status in ("complete", "error", "stopped", "interrupted"):
                yield format_sse_event(
                    kind="stream_end",
                    timestamp=time.time(),
                    payload={"status": snapshot.status, "job_id": job_id},
                )
                return

            time_since_heartbeat += poll_interval
            if time_since_heartbeat >= heartbeat_interval:
                yield format_heartbeat()
                time_since_heartbeat = 0.0

            await asyncio.sleep(poll_interval)

    return EventSourceResponse(event_generator())
