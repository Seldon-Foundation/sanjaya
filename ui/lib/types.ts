/** Types mirroring backend MMOU models and trace events. */

export interface TraceEvent {
  kind: string;
  timestamp: number;
  payload: Record<string, unknown>;
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

export interface MMOUEvaluationSummary {
  answered_accuracy_pct: number;
  correct: number;
  answered: number;
  evaluated_at: string;
  submission_rows: number;
}

export interface MMOUQuestionEvaluationSummary {
  question_id: string;
  answer: string;
  correct: boolean;
  answered_accuracy_pct: number;
  evaluated_at: string;
  submission_rows: number;
}

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
  latest_evaluation: MMOUQuestionEvaluationSummary | null;
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
  latest_evaluation: MMOUEvaluationSummary | null;
}

export interface MMOUQuestionTraceResponse {
  question_id: string;
  run_id: string | null;
  events: TraceEvent[];
}
