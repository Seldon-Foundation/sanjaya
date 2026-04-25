"""Run management endpoints: start runs and stream SSE events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from sanjaya_api.models import DocumentRunRequest, ImageRunRequest, RunRequest, RunResponse
from sanjaya_api.services.orchestrator import OrchestratorService
from sanjaya_api.sse import format_heartbeat, format_sse_event
from sanjaya_api.trace_events import normalize_trace_event

router = APIRouter()

# Singleton orchestrator service (lives for the lifetime of the process)
_orchestrator = OrchestratorService()

@router.post("/runs", response_model=RunResponse)
async def start_run(request: RunRequest) -> RunResponse:
    """Start a new VideoRLM orchestration run."""
    run_id = _orchestrator.start_run(
        video_path=request.video_path,
        question=request.question,
        subtitle_path=request.subtitle_path,
        max_iterations=request.max_iterations,
        max_depth=request.max_depth,
    )
    return RunResponse(run_id=run_id)


@router.post("/runs/image", response_model=RunResponse)
async def start_image_run(request: ImageRunRequest) -> RunResponse:
    """Start a new image analysis run."""
    run_id = _orchestrator.start_image_run(
        image_paths=request.image_paths,
        question=request.question,
        max_iterations=request.max_iterations,
    )
    return RunResponse(run_id=run_id)


@router.post("/runs/document", response_model=RunResponse)
async def start_document_run(request: DocumentRunRequest) -> RunResponse:
    """Start a new document analysis run."""
    run_id = _orchestrator.start_document_run(
        document_paths=request.document_paths,
        question=request.question,
        max_iterations=request.max_iterations,
    )
    return RunResponse(run_id=run_id)


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> EventSourceResponse:
    """Stream trace events for a run via Server-Sent Events."""
    record = _orchestrator.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        cursor = 0
        heartbeat_interval = 2.0  # seconds
        poll_interval = 0.25  # seconds
        time_since_heartbeat = 0.0

        while True:
            tracer = record.tracer
            if tracer is not None:
                events = tracer.events
                # Yield any new events since our cursor
                while cursor < len(events):
                    raw = events[cursor]
                    cursor += 1
                    kind, timestamp, payload = normalize_trace_event(raw)
                    yield format_sse_event(
                        kind=kind,
                        timestamp=timestamp,
                        payload=payload,
                    )

            # Check if run is finished
            if record.status in ("complete", "error"):
                # Drain any remaining events
                if tracer is not None:
                    events = tracer.events
                    while cursor < len(events):
                        raw = events[cursor]
                        cursor += 1
                        kind, timestamp, payload = normalize_trace_event(raw)
                        yield format_sse_event(
                            kind=kind,
                            timestamp=timestamp,
                            payload=payload,
                        )
                # Send terminal status event
                if record.status == "error":
                    yield format_sse_event(
                        kind="stream_error",
                        timestamp=0,
                        payload={"error": record.error or "Unknown error"},
                    )
                yield format_sse_event(
                    kind="stream_end",
                    timestamp=0,
                    payload={"status": record.status},
                )
                return

            # Heartbeat
            time_since_heartbeat += poll_interval
            if time_since_heartbeat >= heartbeat_interval:
                yield format_heartbeat()
                time_since_heartbeat = 0.0

            await asyncio.sleep(poll_interval)

    return EventSourceResponse(event_generator())
