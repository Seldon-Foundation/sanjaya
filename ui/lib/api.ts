/** API client for the Sanjaya MMOU viewer. */

import type {
  MMOUCatalog,
  MMOUEvaluationSummary,
  MMOUJobSummary,
  MMOUQuestionEvaluationSummary,
  MMOUQuestionTraceResponse,
  TraceEvent,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function frameUrl(framePath: string): string {
  return `/api/frames?path=${encodeURIComponent(framePath)}`;
}

export async function fetchMMOUCatalog(): Promise<MMOUCatalog> {
  const res = await fetch(`${API_BASE}/mmou-jobs/catalog`);
  if (!res.ok) {
    throw new Error(`Failed to fetch MMOU catalog: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface MMOUJobCreateRequest {
  benchmark_type?: "mmou";
  limit?: number;
  stratified?: boolean;
  domains?: string[] | null;
  question_ids?: string[] | null;
  workers?: number;
  max_iterations?: number;
  max_depth?: number;
  max_budget_usd?: number | null;
  max_timeout_s?: number | null;
  output_dir?: string | null;
  run_name?: string | null;
  keep_artifacts?: boolean;
}

export async function createMMOUJob(request: MMOUJobCreateRequest): Promise<MMOUJobSummary> {
  const res = await fetch(`${API_BASE}/mmou-jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to create MMOU job: ${res.status}`);
  }
  return res.json();
}

export async function fetchMMOUJobs(): Promise<MMOUJobSummary[]> {
  const res = await fetch(`${API_BASE}/mmou-jobs`);
  if (!res.ok) {
    throw new Error(`Failed to fetch MMOU jobs: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function stopMMOUJob(jobId: string): Promise<MMOUJobSummary> {
  const res = await fetch(`${API_BASE}/mmou-jobs/${encodeURIComponent(jobId)}/stop`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to stop MMOU job: ${res.status}`);
  }
  return res.json();
}

export async function resumeMMOUJob(jobId: string): Promise<MMOUJobSummary> {
  const res = await fetch(`${API_BASE}/mmou-jobs/${encodeURIComponent(jobId)}/resume`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to resume MMOU job: ${res.status}`);
  }
  return res.json();
}

export async function evaluateMMOUJob(jobId: string): Promise<MMOUEvaluationSummary> {
  const res = await fetch(`${API_BASE}/mmou-jobs/${encodeURIComponent(jobId)}/evaluate`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to score MMOU job: ${res.status}`);
  }
  return res.json();
}

export async function evaluateMMOUQuestion(
  jobId: string,
  questionId: string
): Promise<MMOUQuestionEvaluationSummary> {
  const res = await fetch(
    `${API_BASE}/mmou-jobs/${encodeURIComponent(jobId)}/questions/${encodeURIComponent(questionId)}/evaluate`,
    { method: "POST" }
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to score MMOU question: ${res.status}`);
  }
  return res.json();
}

export async function fetchMMOUQuestionTrace(
  jobId: string,
  questionId: string
): Promise<MMOUQuestionTraceResponse> {
  const res = await fetch(
    `${API_BASE}/mmou-jobs/${encodeURIComponent(jobId)}/questions/${encodeURIComponent(questionId)}/trace`
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch MMOU question trace: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function streamMMOUJobEvents(
  jobId: string,
  onUpdate: (job: MMOUJobSummary) => void,
  onTraceEvent: (questionId: string, event: TraceEvent) => void,
  onError: (error: string) => void,
  onEnd: () => void
): () => void {
  const url = `${API_BASE}/mmou-jobs/${jobId}/events`;
  const eventSource = new EventSource(url);

  const handleMessage = (e: MessageEvent) => {
    try {
      const parsed = JSON.parse(e.data) as TraceEvent;
      if (parsed.kind === "stream_end") {
        onEnd();
        eventSource.close();
        return;
      }
      if (parsed.kind === "stream_error") {
        onError((parsed.payload as { error?: string }).error ?? "Unknown error");
        eventSource.close();
        return;
      }
      if (parsed.kind === "mmou_job_update") {
        onUpdate(parsed.payload as unknown as MMOUJobSummary);
        return;
      }
      if (parsed.kind === "mmou_trace_event") {
        const payload = parsed.payload as {
          question_id?: string;
          event?: TraceEvent;
        };
        if (typeof payload.question_id === "string" && payload.event) {
          onTraceEvent(payload.question_id, payload.event);
        }
      }
    } catch {
      // Ignore malformed heartbeats.
    }
  };

  for (const type of ["mmou_job_update", "mmou_trace_event", "stream_end", "stream_error"]) {
    eventSource.addEventListener(type, handleMessage);
  }
  eventSource.onmessage = handleMessage;
  eventSource.onerror = () => {
    onError("MMOU event stream disconnected");
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}
