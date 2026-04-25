"""API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Payload to start a new VideoRLM run."""

    video_path: str
    question: str
    subtitle_path: str | None = None
    max_iterations: int = 20
    max_depth: int = Field(default=1, ge=1, le=8)


class DocumentRunRequest(BaseModel):
    """Payload to start a document analysis run."""

    document_paths: list[str]
    question: str
    max_iterations: int = 12


class ImageRunRequest(BaseModel):
    """Payload to start an image analysis run."""

    image_paths: list[str]
    question: str
    max_iterations: int = 10


class RunResponse(BaseModel):
    """Response after starting a run."""

    run_id: str


class SSEEvent(BaseModel):
    """Shape of a single SSE event delivered to the frontend."""

    kind: str
    timestamp: float
    payload: dict[str, Any]


class BenchmarkPromptCatalogItem(BaseModel):
    """One selectable video benchmark prompt."""

    prompt_id: int
    prompt_name: str
    video_key: str
    question: str
    is_mcq: bool
    group: Literal["demo", "lvb"]


class BenchmarkCatalogResponse(BaseModel):
    """Catalog metadata for benchmark launcher UI."""

    benchmark_type: Literal["video"] = "video"
    prompts: list[BenchmarkPromptCatalogItem]
    defaults: dict[str, Any]


class BenchmarkJobCreateRequest(BaseModel):
    """Payload to start a video benchmark batch job."""

    benchmark_type: Literal["video"] = "video"
    prompt_ids: list[int] | None = None
    workers: int = Field(default=6, ge=1, le=32)
    max_iterations: int = Field(default=20, ge=1, le=100)
    max_depth: int = Field(default=2, ge=1, le=8)
    max_budget_usd: float = Field(default=1.0, gt=0)
    fast: bool = False
    output_dir: str | None = None
    run_name: str | None = None
    download_lvb: bool = False


class BenchmarkPromptTraceResponse(BaseModel):
    """Trace snapshot for one benchmark prompt."""

    prompt_id: int
    run_id: str | None = None
    events: list[SSEEvent]


class BenchmarkPromptStatus(BaseModel):
    """Execution state for one prompt inside a benchmark batch."""

    prompt_id: int
    prompt_name: str
    video_key: str
    question: str
    is_mcq: bool
    group: Literal["demo", "lvb"]
    status: Literal["pending", "running", "complete", "error", "stopped"]
    started_at: float | None = None
    finished_at: float | None = None
    run_id: str | None = None
    result_path: str | None = None
    trace_path: str | None = None
    trace_event_count: int = 0
    iterations: int | None = None
    cost_usd: float | None = None
    wall_time_s: float | None = None
    error: str | None = None
    mcq_correct: bool | None = None


class BenchmarkJobSummary(BaseModel):
    """Job status payload returned to the dashboard."""

    job_id: str
    benchmark_type: Literal["video"] = "video"
    status: Literal["pending", "running", "stopping", "complete", "error", "stopped"]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested_at: float | None = None
    stop_reason: str | None = None
    run_name: str
    output_dir: str
    models: dict[str, str | None]
    workers: int
    max_iterations: int
    max_depth: int
    max_budget_usd: float
    fast: bool
    download_lvb: bool
    total_prompts: int
    completed_prompts: int
    error_prompts: int
    active_prompt_ids: list[int]
    prompt_ids: list[int]
    prompts: list[BenchmarkPromptStatus]
    stdout_tail: list[str]
    stderr_tail: list[str]
    revision: int
