/** API client for the Sanjaya backend. */

import type {
  BenchmarkCatalog,
  BenchmarkJobSummary,
  BenchmarkPromptTraceResponse,
  BenchmarkData,
  MMOUCatalog,
  MMOUEvaluationSummary,
  MMOUJobSummary,
  MMOUQuestionEvaluationSummary,
  MMOUQuestionTraceResponse,
  TraceEvent,
} from "./types";

export interface HistoryEntry {
  runId: string;
  timestamp: number | null;
  videoPath: string | null;
  question: string | null;
  status: "complete" | "error" | "incomplete";
  answerPreview: string | null;
  eventCount: number;
  iterations: number;
  totalTokens: number;
  costUsd: number;
  models: { orchestrator: string | null; recursive: string | null };
}

export async function fetchHistory(): Promise<HistoryEntry[]> {
  const res = await fetch("/api/history");
  if (!res.ok) return [];
  return res.json();
}

export interface RunManifest {
  run_id: string;
  clips?: Record<string, {
    clip_id: string;
    clip_path: string;
    start_s: number;
    end_s: number;
    frame_paths: string[];
    window_id: string;
  }>;
  candidate_windows?: Array<{
    window_id: string;
    strategy: string;
    start_s: number;
    end_s: number;
    score: number;
    reason: string;
  }>;
  media_operations?: Array<{
    kind: string;
    start_s: number;
    end_s: number;
    artifact_path?: string;
    prompt?: string;
    response_preview?: string;
    audio_summary?: string;
  }>;
  trace_events: Array<{
    kind: string;
    timestamp: number;
    payload: Record<string, unknown>;
  }>;
}

export async function fetchRunManifest(runId: string): Promise<RunManifest | null> {
  const res = await fetch(`/api/history/${encodeURIComponent(runId)}`);
  if (!res.ok) return null;
  return res.json();
}

export async function fetchBenchmarks(): Promise<BenchmarkData> {
  const res = await fetch("/api/benchmarks");
  if (!res.ok) throw new Error("Failed to fetch benchmarks");
  return res.json();
}

export async function fetchBenchmarkCatalog(): Promise<BenchmarkCatalog> {
  const res = await fetch(`${API_BASE}/benchmark-jobs/catalog`);
  if (!res.ok) {
    throw new Error(`Failed to fetch benchmark catalog: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface BenchmarkJobCreateRequest {
  benchmark_type?: "video";
  prompt_ids?: number[];
  workers?: number;
  max_iterations?: number;
  max_depth?: number;
  max_budget_usd?: number;
  fast?: boolean;
  output_dir?: string | null;
  run_name?: string | null;
  download_lvb?: boolean;
}

export async function createBenchmarkJob(
  request: BenchmarkJobCreateRequest
): Promise<BenchmarkJobSummary> {
  const res = await fetch(`${API_BASE}/benchmark-jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to create benchmark job: ${res.status}`);
  }
  return res.json();
}

export async function fetchBenchmarkJobs(): Promise<BenchmarkJobSummary[]> {
  const res = await fetch(`${API_BASE}/benchmark-jobs`);
  if (!res.ok) {
    throw new Error(`Failed to fetch benchmark jobs: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function stopBenchmarkJob(jobId: string): Promise<BenchmarkJobSummary> {
  const res = await fetch(`${API_BASE}/benchmark-jobs/${encodeURIComponent(jobId)}/stop`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Failed to stop benchmark job: ${res.status}`);
  }
  return res.json();
}

export async function fetchBenchmarkPromptTrace(
  jobId: string,
  promptId: number
): Promise<BenchmarkPromptTraceResponse> {
  const res = await fetch(`${API_BASE}/benchmark-jobs/${encodeURIComponent(jobId)}/prompts/${promptId}/trace`);
  if (!res.ok) {
    throw new Error(`Failed to fetch prompt trace: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function fetchDocumentBenchmarks(): Promise<import("./types").DocumentBenchmarkData> {
  const res = await fetch("/api/document-benchmarks");
  if (!res.ok) throw new Error("Failed to fetch document benchmarks");
  return res.json();
}

export async function fetchImageBenchmarks(): Promise<import("./types").ImageBenchmarkData> {
  const res = await fetch("/api/image-benchmarks");
  if (!res.ok) throw new Error("Failed to fetch image benchmarks");
  return res.json();
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface RunRequest {
  video_path: string;
  question: string;
  subtitle_path?: string;
  max_iterations?: number;
  max_depth?: number;
}

export interface VideoEntry {
  path: string;
  hasTranscript: boolean;
}

export async function fetchVideos(): Promise<VideoEntry[]> {
  const res = await fetch("/api/videos");
  if (!res.ok) {
    throw new Error(`Failed to fetch videos: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export async function fetchTranscript(
  videoRelPath: string
): Promise<TranscriptSegment[]> {
  const res = await fetch(`/api/videos/transcript?path=${encodeURIComponent(videoRelPath)}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.segments ?? [];
}

export function videoStreamUrl(videoRelPath: string): string {
  return `/api/videos/stream?path=${encodeURIComponent(videoRelPath)}`;
}

export function frameUrl(framePath: string): string {
  return `/api/frames?path=${encodeURIComponent(framePath)}`;
}

function traceEventKey(event: TraceEvent): string {
  return `${event.kind}:${event.timestamp}:${JSON.stringify(event.payload)}`;
}

export async function submitRun(
  request: RunRequest
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    throw new Error(`Failed to start run: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface DocumentRunRequest {
  document_paths: string[];
  question: string;
  max_iterations?: number;
}

export async function uploadDocuments(
  files: File[]
): Promise<{ paths: string[]; count: number }> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const res = await fetch("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export interface ImageRunRequest {
  image_paths: string[];
  question: string;
  max_iterations?: number;
}

export async function uploadImages(
  files: File[]
): Promise<{ paths: string[]; count: number }> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const res = await fetch("/api/images/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function submitImageRun(
  request: ImageRunRequest
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs/image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    throw new Error(`Failed to start image run: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function submitDocumentRun(
  request: DocumentRunRequest
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs/document`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    throw new Error(`Failed to start document run: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function streamEvents(
  runId: string,
  onEvent: (event: TraceEvent) => void,
  onError: (error: string) => void,
  onEnd: () => void
): () => void {
  const url = `${API_BASE}/runs/${runId}/events`;
  const eventSource = new EventSource(url);
  const seenKeys = new Set<string>();

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
      const key = traceEventKey(parsed);
      if (seenKeys.has(key)) return;
      seenKeys.add(key);
      onEvent(parsed);
    } catch {
      // ignore parse errors on heartbeats
    }
  };

  // Listen to all named event types we care about.
  // Includes both legacy names and mapped names from the backend _KIND_MAP.
  const eventTypes = [
    "run_start",
    "run_end",
    "root_response",
    "root_response_start",
    "code_instruction",
    "code_execution",
    "video_inspection_start",
    "video_inspection",
    "frame_inspection_start",
    "frame_inspection",
    "audio_analysis_start",
    "audio_analysis",
    "vision",
    "sub_llm",
    "sub_llm_start",
    "subcall",
    "subcall_start",
    "image_inspection",
    "image_compare",
    "iteration_start",
    "iteration_end",
    "tool_call",
    "tool_call_start",
    "schema_generation",
    "schema_generation_start",
    "critic_evaluation",
    "heartbeat",
    "stream_end",
    "stream_error",
  ];

  for (const type of eventTypes) {
    eventSource.addEventListener(type, handleMessage);
  }

  // Also listen to generic messages as fallback
  eventSource.onmessage = handleMessage;

  eventSource.onerror = () => {
    onError("SSE connection error");
    eventSource.close();
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

export function streamBenchmarkJobEvents(
  jobId: string,
  onUpdate: (job: BenchmarkJobSummary) => void,
  onTraceEvent: (promptId: number, event: TraceEvent) => void,
  onError: (error: string) => void,
  onEnd: () => void
): () => void {
  const url = `${API_BASE}/benchmark-jobs/${jobId}/events`;
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
      if (parsed.kind === "benchmark_job_update") {
        onUpdate(parsed.payload as unknown as BenchmarkJobSummary);
        return;
      }
      if (parsed.kind === "benchmark_trace_event") {
        const payload = parsed.payload as {
          prompt_id?: number;
          event?: TraceEvent;
        };
        if (typeof payload.prompt_id === "number" && payload.event) {
          onTraceEvent(payload.prompt_id, payload.event);
        }
      }
    } catch {
      // ignore parse errors on heartbeats
    }
  };

  const eventTypes = [
    "benchmark_job_update",
    "benchmark_trace_event",
    "stream_end",
    "stream_error",
  ];
  for (const type of eventTypes) {
    eventSource.addEventListener(type, handleMessage);
  }
  eventSource.onmessage = handleMessage;
  eventSource.onerror = () => {
    onError("Benchmark event stream disconnected");
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
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
      // ignore parse errors on heartbeats
    }
  };

  const eventTypes = [
    "mmou_job_update",
    "mmou_trace_event",
    "stream_end",
    "stream_error",
  ];
  for (const type of eventTypes) {
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
