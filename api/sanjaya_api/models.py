"""API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SSEEvent(BaseModel):
    """Shape of a single SSE event delivered to the frontend."""

    kind: str
    timestamp: float
    payload: dict[str, Any]


class MMOUCatalogResponse(BaseModel):
    """Catalog metadata for the MMOU benchmark launcher UI."""

    benchmark_type: Literal["mmou"] = "mmou"
    total_questions: int
    domain_counts: dict[str, int]
    defaults: dict[str, Any]


class MMOUJobCreateRequest(BaseModel):
    """Payload to start an MMOU benchmark job."""

    benchmark_type: Literal["mmou"] = "mmou"
    limit: int = Field(default=10, ge=1, le=5000)
    stratified: bool = True
    domains: list[str] | None = None
    question_ids: list[str] | None = None
    workers: int = Field(default=1, ge=1, le=16)
    max_iterations: int = Field(default=20, ge=1, le=100)
    max_depth: int = Field(default=2, ge=1, le=8)
    max_budget_usd: float | None = Field(default=None, gt=0)
    max_timeout_s: float | None = Field(default=None, gt=0)
    output_dir: str | None = None
    run_name: str | None = None
    keep_artifacts: bool = False
    benchmarks_dir: str | None = None
    dataset_file: str | None = None


class MMOUQuestionEvaluationSummary(BaseModel):
    """Compact MMOU evaluator result for one answered question."""

    question_id: str
    answer: str
    correct: bool
    answered_accuracy_pct: float
    evaluated_at: str
    submission_rows: int


class MMOUQuestionStatus(BaseModel):
    """Execution state for one MMOU question."""

    question_id: str
    question: str
    options: dict[str, str]
    domain: str
    subdomain: str
    question_type: list[str]
    start_time: str
    end_time: str
    status: Literal["pending", "running", "complete", "error", "stopped"]
    started_at: float | None = None
    finished_at: float | None = None
    run_id: str | None = None
    result_path: str | None = None
    trace_path: str | None = None
    trace_event_count: int = 0
    answer: str | None = None
    raw_text: str | None = None
    parse_error: str | None = None
    attempts: int | None = None
    iterations: int | None = None
    cost_usd: float | None = None
    wall_time_s: float | None = None
    error: str | None = None
    latest_evaluation: MMOUQuestionEvaluationSummary | None = None


class MMOUEvaluationSummary(BaseModel):
    """Compact MMOU evaluator result for answered questions only."""

    answered_accuracy_pct: float
    correct: int
    answered: int
    evaluated_at: str
    submission_rows: int


class MMOUJobSummary(BaseModel):
    """MMOU job status payload returned to the dashboard."""

    job_id: str
    benchmark_type: Literal["mmou"] = "mmou"
    status: Literal["pending", "running", "stopping", "complete", "error", "stopped", "interrupted"]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested_at: float | None = None
    stop_reason: str | None = None
    run_name: str
    output_dir: str
    job_dir: str
    models: dict[str, str | None]
    workers: int
    max_iterations: int
    max_depth: int
    max_budget_usd: float | None
    max_timeout_s: float | None
    limit: int
    stratified: bool
    domains: list[str] | None
    selection_source: Literal["dataset", "question_ids"]
    keep_artifacts: bool
    total_questions: int
    completed_questions: int
    error_questions: int
    active_question_ids: list[str]
    question_ids: list[str]
    questions: list[MMOUQuestionStatus]
    stdout_tail: list[str]
    stderr_tail: list[str]
    revision: int
    latest_evaluation: MMOUEvaluationSummary | None = None


class MMOUQuestionTraceResponse(BaseModel):
    """Trace snapshot for one MMOU question."""

    question_id: str
    run_id: str | None = None
    events: list[SSEEvent]
