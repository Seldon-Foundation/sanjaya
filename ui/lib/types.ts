/** Types mirroring backend models and derived UI state. */

export interface TraceEvent {
  kind: string;
  timestamp: number;
  payload: Record<string, unknown>;
}

export type RunStatus = "idle" | "running" | "complete" | "error";

export interface RunState {
  runId: string | null;
  status: RunStatus;
  events: TraceEvent[];
  startTime: number | null;
  currentIteration: number;
  maxIterations: number;
  finalAnswer: string | null;
  finalStatus: string | null;
  error: string | null;
  orchestratorModel: string | null;
  recursiveModel: string | null;
}

/** Derived token/cost totals. */
export interface TokenTotals {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  costUsd: number;
}

/** A code execution entry. */
export interface CodeExecution {
  iteration: number;
  codeBlockIndex: number;
  codeBlockTotal: number;
  code: string;
  executionTime: number;
  stderr: string;
  hasFinalAnswer: boolean;
}

/** A sub-LLM call entry. */
export interface SubLLMCall {
  promptPreview: string;
  responsePreview: string;
  inputTokens: number | null;
  outputTokens: number | null;
  costUsd: number | null;
  modelUsed: string | null;
  durationSeconds: number | null;
}

/** Media operation data from trace events. */
export interface MediaOperationEntry {
  kind: "video_inspection" | "frame_inspection" | "audio_analysis";
  startS: number;
  endS: number;
  prompt: string;
  artifactPath: string | null;
  responsePreview: string;
  inputTokens: number | null;
  outputTokens: number | null;
  costUsd: number | null;
  modelUsed: string | null;
  durationSeconds: number | null;
}

/** Legacy live-run clip entry kept for compatibility with older HUD panels. */
export interface ClipEntry {
  clipId: string;
  startS: number;
  endS: number;
  clipPath: string;
  frameCount: number;
}

/** Legacy vision entry kept for compatibility with older HUD panels. */
export interface VisionEntry {
  prompt: string;
  frameCount: number;
  clipCount: number;
  responsePreview: string;
  inputTokens: number | null;
  outputTokens: number | null;
  costUsd: number | null;
  modelUsed: string | null;
  durationSeconds: number | null;
  framePaths: string[];
  clipId: string | null;
}

/** Evidence item from final answer. */
export interface EvidenceItem {
  windowId: string | null;
  startS: number;
  endS: number;
  rationale: string;
}

/** A single prompt result from one version. */
export interface PromptResult {
  promptId: number;
  promptName: string;
  videoKey: string;
  question: string;
  answerText: string;
  answerData: Record<string, unknown> | null;
  iterations: number;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  wallTimeS: number;
  evidenceCount: number;
  evidenceSources: string[];
  subtitle: {
    hadExistingSubtitle: boolean;
    subtitleGenerated: boolean;
    subtitleSource: string;
  };
  error?: string;
  traceEvents?: TraceEvent[] | null;
}

/** Merged prompt with all available versions. */
export interface BenchmarkPrompt {
  promptId: number;
  promptName: string;
  videoKey: string;
  videoPath: string | null;
  question: string;
  versions: Record<string, PromptResult>;
  bestVersion: string;
  groundTruth?: string | null;
}

export interface LiveRunItem {
  runId: string;
  timestamp: string;
  model: string;
  prompt: BenchmarkPrompt;
}

export interface LiveRunsData {
  runs: LiveRunItem[];
  totalRuns: number;
  totalCostUsd: number;
  totalWallTimeS: number;
}

export interface VideoInfo {
  key: string;
  path: string;
  title: string;
  channel: string;
  youtubeUrl: string;
  duration: string;
}

export interface BenchmarkData {
  prompts: BenchmarkPrompt[];
  summary: {
    totalPrompts: number;
    versions: string[];
    totalCostUsd: number;
    totalWallTimeS: number;
    v1CostUsd: number;
    v1WallTimeS: number;
    latestVersion: string;
    models?: string[];
  };
  liveRuns: LiveRunsData;
  videos: VideoInfo[];
}

export interface BenchmarkPromptCatalogItem {
  prompt_id: number;
  prompt_name: string;
  video_key: string;
  question: string;
  is_mcq: boolean;
  group: "demo" | "lvb";
}

export interface BenchmarkCatalog {
  benchmark_type: "video";
  prompts: BenchmarkPromptCatalogItem[];
  defaults: {
    workers: number;
    max_iterations: number;
    max_depth: number;
    max_budget_usd: number;
    fast: boolean;
    download_lvb: boolean;
    output_dir: string;
    models: Record<string, string | null>;
    prompt_presets: Record<"all" | "demo" | "lvb", number[]>;
  };
}

export type BenchmarkJobStatus = "pending" | "running" | "stopping" | "complete" | "error" | "stopped";
export type BenchmarkPromptJobStatus = "pending" | "running" | "complete" | "error" | "stopped";

export interface BenchmarkJobPrompt {
  prompt_id: number;
  prompt_name: string;
  video_key: string;
  question: string;
  is_mcq: boolean;
  group: "demo" | "lvb";
  status: BenchmarkPromptJobStatus;
  started_at: number | null;
  finished_at: number | null;
  run_id: string | null;
  result_path: string | null;
  trace_path: string | null;
  trace_event_count: number;
  iterations: number | null;
  cost_usd: number | null;
  wall_time_s: number | null;
  error: string | null;
  mcq_correct: boolean | null;
}

export interface BenchmarkJobSummary {
  job_id: string;
  benchmark_type: "video";
  status: BenchmarkJobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  stop_requested_at: number | null;
  stop_reason: string | null;
  run_name: string;
  output_dir: string;
  models: Record<string, string | null>;
  workers: number;
  max_iterations: number;
  max_depth: number;
  max_budget_usd: number;
  fast: boolean;
  download_lvb: boolean;
  total_prompts: number;
  completed_prompts: number;
  error_prompts: number;
  active_prompt_ids: number[];
  prompt_ids: number[];
  prompts: BenchmarkJobPrompt[];
  stdout_tail: string[];
  stderr_tail: string[];
  revision: number;
}

export interface BenchmarkPromptTraceResponse {
  prompt_id: number;
  run_id: string | null;
  events: TraceEvent[];
}

export interface MMOUCatalog {
  benchmark_type: "mmou";
  total_questions: number;
  domain_counts: Record<string, number>;
  defaults: {
    limit: number;
    stratified: boolean;
    workers: number;
    max_iterations: number;
    max_depth: number;
    max_budget_usd: number | null;
    max_timeout_s: number | null;
    output_dir: string;
    keep_artifacts: boolean;
    models: Record<string, string | null>;
  };
}

export type MMOUJobStatus =
  | "pending"
  | "running"
  | "stopping"
  | "complete"
  | "error"
  | "stopped"
  | "interrupted";

export type MMOUQuestionJobStatus = "pending" | "running" | "complete" | "error" | "stopped";

export interface MMOUJobQuestion {
  question_id: string;
  question: string;
  options: Record<string, string>;
  domain: string;
  subdomain: string;
  question_type: string[];
  start_time: string;
  end_time: string;
  status: MMOUQuestionJobStatus;
  started_at: number | null;
  finished_at: number | null;
  run_id: string | null;
  result_path: string | null;
  trace_path: string | null;
  trace_event_count: number;
  answer: string | null;
  raw_text: string | null;
  parse_error: string | null;
  attempts: number | null;
  iterations: number | null;
  cost_usd: number | null;
  wall_time_s: number | null;
  error: string | null;
}

export interface MMOUJobSummary {
  job_id: string;
  benchmark_type: "mmou";
  status: MMOUJobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  stop_requested_at: number | null;
  stop_reason: string | null;
  run_name: string;
  output_dir: string;
  job_dir: string;
  models: Record<string, string | null>;
  workers: number;
  max_iterations: number;
  max_depth: number;
  max_budget_usd: number | null;
  max_timeout_s: number | null;
  limit: number;
  stratified: boolean;
  domains: string[] | null;
  selection_source: "dataset" | "question_ids";
  keep_artifacts: boolean;
  total_questions: number;
  completed_questions: number;
  error_questions: number;
  active_question_ids: string[];
  question_ids: string[];
  questions: MMOUJobQuestion[];
  stdout_tail: string[];
  stderr_tail: string[];
  revision: number;
}

export interface MMOUQuestionTraceResponse {
  question_id: string;
  run_id: string | null;
  events: TraceEvent[];
}

/* ── Document benchmark types ─────────────────────────── */

export interface DocumentResult {
  promptId: number;
  promptName: string;
  collection: string;
  documentPaths: string[];
  question: string;
  answerText: string;
  answerData: Record<string, unknown> | null;
  iterations: number;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  wallTimeS: number;
  evidenceCount: number;
  evidenceSources: string[];
  error?: string;
  traceEvents?: TraceEvent[] | null;
}

export interface DocumentPrompt {
  promptId: number;
  promptName: string;
  collection: string;
  question: string;
  versions: Record<string, DocumentResult>;
  bestVersion: string;
}

export interface DocumentSource {
  name: string;
  type: string;
}

export interface DocumentBenchmarkData {
  prompts: DocumentPrompt[];
  summary: {
    totalPrompts: number;
    versions: string[];
    totalCostUsd: number;
    totalWallTimeS: number;
    latestVersion: string;
  };
  documents: DocumentSource[];
}

/* ── Image benchmark types ────────────────────────────── */

export interface MuirbenchChoice {
  label: string;
  text: string;
  isImageRef: boolean;
  imagePath: string | null;
  imageTag: string | null;
}

export interface MuirbenchMeta {
  idx: string;
  answer: string | null;
  choices: MuirbenchChoice[];
}

export interface ImageResult {
  promptId: number;
  promptName: string;
  imagePaths: string[];
  question: string;
  answerText: string;
  answerData: Record<string, unknown> | null;
  groundTruth: string | null;
  iterations: number;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  wallTimeS: number;
  evidenceCount: number;
  evidenceSources: string[];
  error?: string;
  traceEvents?: TraceEvent[] | null;
  muirbench?: MuirbenchMeta | null;
}

export interface ImagePrompt {
  promptId: number;
  promptName: string;
  question: string;
  versions: Record<string, ImageResult>;
  bestVersion: string;
}

export interface BenchmarkAccuracy {
  correct: number;
  total: number;
  accuracy: number;
}

export interface ImageBenchmarkData {
  prompts: ImagePrompt[];
  summary: {
    totalPrompts: number;
    versions: string[];
    totalCostUsd: number;
    totalWallTimeS: number;
    latestVersion: string;
    models?: string[];
    benchmarks?: string[];
    benchmarkSummaries?: Record<string, BenchmarkAccuracy>;
  };
}
