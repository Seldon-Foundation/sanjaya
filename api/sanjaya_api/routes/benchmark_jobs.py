"""Video benchmark batch job endpoints."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from sanjaya_api.models import (
    BenchmarkCatalogResponse,
    BenchmarkJobCreateRequest,
    BenchmarkJobSummary,
    BenchmarkPromptTraceResponse,
)
from sanjaya_api.services.benchmark_jobs import BenchmarkJobService
from sanjaya_api.sse import format_heartbeat, format_sse_event
from sanjaya_api.trace_events import normalize_trace_event

router = APIRouter()

_benchmark_jobs = BenchmarkJobService()


@router.get("/benchmark-jobs/catalog", response_model=BenchmarkCatalogResponse)
async def get_benchmark_catalog() -> BenchmarkCatalogResponse:
    """Return prompt metadata and defaults for the video benchmark launcher."""
    return _benchmark_jobs.get_catalog()


@router.post("/benchmark-jobs", response_model=BenchmarkJobSummary)
async def create_benchmark_job(request: BenchmarkJobCreateRequest) -> BenchmarkJobSummary:
    """Create a new benchmark batch job."""
    try:
        return _benchmark_jobs.start_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/benchmark-jobs", response_model=list[BenchmarkJobSummary])
async def list_benchmark_jobs() -> list[BenchmarkJobSummary]:
    """List benchmark jobs newest-first."""
    return _benchmark_jobs.list_jobs()


@router.get("/benchmark-jobs/{job_id}", response_model=BenchmarkJobSummary)
async def get_benchmark_job(job_id: str) -> BenchmarkJobSummary:
    """Fetch one benchmark job snapshot."""
    job = _benchmark_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Benchmark job {job_id} not found")
    return job


@router.post("/benchmark-jobs/{job_id}/stop", response_model=BenchmarkJobSummary)
async def stop_benchmark_job(job_id: str) -> BenchmarkJobSummary:
    """Request cooperative stop for a benchmark job."""
    job = _benchmark_jobs.request_stop(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Benchmark job {job_id} not found")
    return job


@router.get("/benchmark-jobs/{job_id}/prompts/{prompt_id}/trace", response_model=BenchmarkPromptTraceResponse)
async def get_prompt_trace(job_id: str, prompt_id: int) -> BenchmarkPromptTraceResponse:
    """Fetch the current trace snapshot for one prompt in a benchmark job."""
    result = _benchmark_jobs.get_prompt_trace(job_id, prompt_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Trace for prompt {prompt_id} not found")
    run_id, events = result
    return BenchmarkPromptTraceResponse(prompt_id=prompt_id, run_id=run_id, events=events)


@router.get("/benchmark-jobs/{job_id}/events")
async def stream_benchmark_job_events(job_id: str) -> EventSourceResponse:
    """Stream benchmark job updates and prompt trace events."""
    record = _benchmark_jobs.get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Benchmark job {job_id} not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        revision = -1
        cursors: dict[int, int] = {}
        heartbeat_interval = 2.0
        poll_interval = 0.25
        time_since_heartbeat = 0.0

        while True:
            current = _benchmark_jobs.get_job_record(job_id)
            if current is None:
                yield format_sse_event(
                    kind="stream_error",
                    timestamp=time.time(),
                    payload={"error": f"Benchmark job {job_id} disappeared"},
                )
                return

            snapshot = _benchmark_jobs.get_job(job_id)
            if snapshot is None:
                yield format_sse_event(
                    kind="stream_error",
                    timestamp=time.time(),
                    payload={"error": f"Benchmark job {job_id} disappeared"},
                )
                return

            if snapshot.revision != revision:
                revision = snapshot.revision
                yield format_sse_event(
                    kind="benchmark_job_update",
                    timestamp=time.time(),
                    payload=snapshot.model_dump(),
                )

            for prompt_id in current.prompt_ids:
                prompt_record = current.prompts[prompt_id]
                tracer = prompt_record.tracer
                if tracer is None:
                    continue
                cursor = cursors.get(prompt_id, 0)
                events = tracer.events
                while cursor < len(events):
                    raw = events[cursor]
                    cursor += 1
                    kind, timestamp, payload = normalize_trace_event(raw)
                    yield format_sse_event(
                        kind="benchmark_trace_event",
                        timestamp=timestamp,
                        payload={
                            "job_id": job_id,
                            "prompt_id": prompt_id,
                            "event": {
                                "kind": kind,
                                "timestamp": timestamp,
                                "payload": payload,
                            },
                        },
                    )
                cursors[prompt_id] = cursor

            if snapshot.status in ("complete", "error", "stopped"):
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
