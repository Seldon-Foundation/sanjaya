export interface TraceEventLike {
  kind: string;
  timestamp: number;
  payload?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface NormalizedTraceEvent {
  kind: string;
  timestamp: number;
  payload: Record<string, unknown>;
}

export const TRACE_KIND_MAP: Record<string, string> = {
  "sanjaya.completion_start": "run_start",
  "sanjaya.completion_end": "run_end",
  "sanjaya.iteration_start": "iteration_start",
  "sanjaya.iteration_end": "iteration_end",
  "sanjaya.root_llm_call_start": "root_response_start",
  "sanjaya.root_llm_call_end": "root_response",
  "sanjaya.code_execution_start": "code_instruction",
  "sanjaya.code_execution_end": "code_execution",
  "sanjaya.tool_call_start": "tool_call_start",
  "sanjaya.tool_call_end": "tool_call",
  "sanjaya.sub_llm_call.regular_start": "sub_llm_start",
  "sanjaya.sub_llm_call.regular_end": "sub_llm",
  "sanjaya.rlm_subcall_start": "subcall_start",
  "sanjaya.rlm_subcall_end": "subcall",
  "sanjaya.sub_llm_call.vision_start": "vision_start",
  "sanjaya.sub_llm_call.vision_end": "vision",
  "sanjaya.video_inspection_start": "video_inspection_start",
  "sanjaya.video_inspection_end": "video_inspection",
  "sanjaya.frame_inspection_start": "frame_inspection_start",
  "sanjaya.frame_inspection_end": "frame_inspection",
  "sanjaya.audio_analysis_start": "audio_analysis_start",
  "sanjaya.audio_analysis_end": "audio_analysis",
  "sanjaya.schema_generation_start": "schema_generation_start",
  "sanjaya.schema_generation_end": "schema_generation",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function normalizeTraceEvents(events: TraceEventLike[]): NormalizedTraceEvent[] {
  return events.map((event) => {
    const nestedPayload = isRecord(event.payload) ? event.payload : {};
    const topLevelPayload = Object.fromEntries(
      Object.entries(event).filter(([key]) => key !== "kind" && key !== "timestamp" && key !== "payload"),
    );

    return {
      kind: TRACE_KIND_MAP[event.kind] ?? event.kind,
      timestamp: event.timestamp,
      payload: { ...topLevelPayload, ...nestedPayload },
    };
  });
}
