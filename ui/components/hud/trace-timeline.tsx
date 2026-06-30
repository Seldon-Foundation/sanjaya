"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { Panel } from "./panel";
import { frameUrl } from "@/lib/api";
import type { TraceEvent } from "@/lib/types";

interface TraceTimelineProps {
  events: TraceEvent[];
  startTime: number | null;
}

function getEventColor(kind: string): string {
  switch (kind) {
    case "run_start":
    case "run_end":
      return "text-foreground";
    case "root_response":
      return "text-hud-blue";
    case "code_execution":
    case "code_instruction":
      return "text-hud-green";
    case "iteration_end":
      return "text-hud-amber";
    case "vision":
      return "text-hud-magenta";
    case "subcall_start":
    case "subcall":
      return "text-hud-green";
    case "sub_llm":
      return "text-hud-dim";
    case "tool_call":
    case "frame_inspection":
      return "text-hud-cyan";
    case "video_inspection":
      return "text-hud-magenta";
    case "audio_analysis":
    case "schema_generation":
      return "text-hud-label";
    default:
      return "text-hud-dim";
  }
}

function getDotColor(kind: string): string {
  switch (kind) {
    case "run_start":
    case "run_end":
      return "bg-foreground";
    case "root_response":
      return "bg-hud-blue";
    case "code_execution":
    case "code_instruction":
      return "bg-hud-green";
    case "iteration_end":
      return "bg-hud-amber";
    case "vision":
      return "bg-hud-magenta";
    case "subcall_start":
    case "subcall":
      return "bg-hud-green";
    case "sub_llm":
      return "bg-hud-dim";
    case "tool_call":
    case "frame_inspection":
      return "bg-hud-cyan";
    case "video_inspection":
      return "bg-hud-magenta";
    case "audio_analysis":
    case "schema_generation":
      return "bg-hud-label";
    default:
      return "bg-hud-dim";
  }
}

function explicitEventDepth(event: TraceEvent): number | null {
  const depth = event.payload?.depth;
  return typeof depth === "number" && Number.isFinite(depth) ? Math.max(0, depth) : null;
}

function mediaSource(event: TraceEvent): string {
  const source = event.payload?.source;
  return typeof source === "string" ? source : "";
}

function inferredEventDepth(event: TraceEvent, currentSubcallDepth: number): number {
  const explicitDepth = explicitEventDepth(event);
  if (explicitDepth != null) {
    return explicitDepth;
  }

  const source = mediaSource(event);
  if (
    event.kind === "root_response" ||
    event.kind === "code_instruction" ||
    event.kind === "code_execution" ||
    event.kind === "iteration_end"
  ) {
    return currentSubcallDepth;
  }
  if (event.kind === "subcall_start" || event.kind === "subcall") {
    return Math.max(1, currentSubcallDepth || 1);
  }
  if (source === "inspect_video") {
    return currentSubcallDepth;
  }
  if (source === "llm_query" || source === "llm_query_batched" || source === "rlm_query_leaf") {
    return currentSubcallDepth + 1;
  }
  if (event.kind === "sub_llm" || event.kind === "vision") {
    return currentSubcallDepth + 1;
  }
  return 0;
}

function deriveDisplayDepths(events: TraceEvent[]): number[] {
  const stack: number[] = [];
  let topRootModel: string | null = null;
  let recursiveModel: string | null = null;
  let implicitRootDepth = 0;

  return events.map((event) => {
    const explicitDepth = explicitEventDepth(event);
    const currentSubcallDepth = stack[stack.length - 1] ?? 0;
    const payloadModel =
      typeof event.payload?.model === "string"
        ? event.payload.model
        : typeof event.payload?.model_used === "string"
          ? event.payload.model_used
          : typeof event.payload?.child_model === "string"
            ? event.payload.child_model
            : null;

    if (event.kind === "run_start") {
      topRootModel =
        typeof event.payload?.orchestrator_model === "string"
          ? event.payload.orchestrator_model
          : typeof event.payload?.model === "string"
            ? event.payload.model
            : null;
      recursiveModel =
        typeof event.payload?.recursive_model === "string"
          ? event.payload.recursive_model
          : null;
      implicitRootDepth = 0;
    }

    if (event.kind === "subcall_start") {
      const depth = explicitDepth ?? Math.max(1, currentSubcallDepth + 1);
      stack.push(depth);
       implicitRootDepth = depth;
      return depth;
    }

    if (event.kind === "subcall") {
      const depth = explicitDepth ?? Math.max(1, currentSubcallDepth || 1);
      if (stack.length > 0) {
        stack.pop();
      }
      implicitRootDepth = stack[stack.length - 1] ?? 0;
      return depth;
    }

    if (event.kind === "root_response" && explicitDepth == null) {
      if (currentSubcallDepth > 0) {
        implicitRootDepth = currentSubcallDepth;
        return currentSubcallDepth;
      }
      if (
        payloadModel &&
        recursiveModel &&
        payloadModel === recursiveModel &&
        payloadModel !== topRootModel
      ) {
        implicitRootDepth = 1;
        return 1;
      }
      implicitRootDepth = 0;
      return 0;
    }

    if (
      explicitDepth == null &&
      (event.kind === "code_instruction" || event.kind === "code_execution" || event.kind === "iteration_end")
    ) {
      if (currentSubcallDepth > 0) {
        implicitRootDepth = currentSubcallDepth;
        return currentSubcallDepth;
      }
      return implicitRootDepth;
    }

    return inferredEventDepth(event, currentSubcallDepth);
  });
}

function mediaEventLabel(event: TraceEvent): string {
  const source = mediaSource(event);
  switch (source) {
    case "llm_query":
      if (event.kind === "frame_inspection") return "LLM Frame Query";
      if (event.kind === "audio_analysis") return "LLM Audio Query";
      return "LLM Video Query";
    case "llm_query_batched":
      if (event.kind === "frame_inspection") return "LLM Frame Batch";
      if (event.kind === "audio_analysis") return "LLM Audio Batch";
      return "LLM Video Batch";
    case "rlm_query_leaf":
      if (event.kind === "frame_inspection") return "RLM Leaf Frame";
      if (event.kind === "audio_analysis") return "RLM Leaf Audio";
      return "RLM Leaf Video";
    default:
      if (event.kind === "frame_inspection") return "Inspect Frame";
      if (event.kind === "audio_analysis") return "Analyze Audio";
      return "Inspect Video";
  }
}

function eventLabel(event: TraceEvent, depth: number): string {
  switch (event.kind) {
    case "run_start":
      return "Run Start";
    case "run_end":
      return "Run End";
    case "iteration_end":
      return depth > 0 ? "RLM Iter" : "Iteration";
    case "root_response":
      return depth > 0 ? "RLM Root" : "Root LLM";
    case "code_instruction":
      return "Code Plan";
    case "code_execution":
      return "Code Exec";
    case "subcall_start":
      return "RLM Enter";
    case "subcall":
      return "RLM Subcall";
    case "sub_llm":
      return "Sub LLM";
    case "tool_call":
      return "Tool Call";
    case "video_inspection":
    case "frame_inspection":
    case "audio_analysis":
      return mediaEventLabel(event);
    case "schema_generation":
      return "Schema";
    case "vision":
      return "Vision";
    default:
      return event.kind.replaceAll("_", " ");
  }
}

function shortModelName(model: unknown): string {
  if (typeof model !== "string" || model.length === 0) return "?";
  const parts = model.split("/");
  return parts[parts.length - 1] || model;
}

function depthBadgeLabel(depth: number): string {
  return depth <= 0 ? "ROOT" : `D${depth}`;
}

function depthBadgeClass(depth: number): string {
  if (depth <= 0) {
    return "border-hud-border text-hud-dim";
  }
  if (depth === 1) {
    return "border-hud-blue/50 text-hud-blue";
  }
  if (depth === 2) {
    return "border-hud-magenta/50 text-hud-magenta";
  }
  return "border-hud-amber/50 text-hud-amber";
}

function getEventSummary(event: TraceEvent): string {
  const p = event.payload;
  if (!p) return "";
  const source = mediaSource(event);

  try {
    switch (event.kind) {
      case "run_start":
        return `${shortModelName((p.model as string) ?? (p.orchestrator_model as string) ?? "")}${p.question ? ` · "${(p.question as string).slice(0, 72)}"` : ""}`;
      case "run_end":
        return `status=${p.status ?? "complete"} tokens=${p.input_tokens ?? "?"}/${p.output_tokens ?? "?"}`;
      case "iteration_end":
        return `iteration ${p.iteration ?? "?"}${p.final_answer ? " · final" : ""}`;
      case "root_response":
        return `${shortModelName(p.model)}${p.attached_media_count ? ` · +${p.attached_media_count} media` : ""}${p.input_tokens ? ` · in ${p.input_tokens}` : ""}${p.output_tokens ? ` · out ${p.output_tokens}` : ""}${p.response_preview ? ` · ${(p.response_preview as string).slice(0, 70)}` : ""}`;
      case "code_instruction":
        return `${((p.code_preview as string) ?? "").slice(0, 72)}`;
      case "code_execution":
        return `block ${p.block_index ?? "?"} · ${(p.execution_time_s as number)?.toFixed(2) ?? (p.execution_time as number)?.toFixed(2) ?? "?"}s${(p.tools_used as string[])?.length ? ` · ${(p.tools_used as string[])?.join(", ")}` : ""}${p.final_answer ? " · final" : ""}`;
      case "tool_call":
        return `${p.tool_name ?? "?"}`;
      case "video_inspection":
      case "frame_inspection":
      case "audio_analysis":
        return `${source ? `${source} · ` : ""}[${(p.start_s as number)?.toFixed(1) ?? "?"}s - ${(p.end_s as number)?.toFixed(1) ?? "?"}s]${p.model_used ? ` · ${shortModelName(p.model_used)}` : (p.model as string) ? ` · ${shortModelName(p.model)}` : ""}${p.response_preview ? ` · ${(p.response_preview as string).slice(0, 70)}` : ""}`;
      case "subcall_start":
        return `${shortModelName(p.child_model)}${p.start_s != null && p.end_s != null ? ` · [${(p.start_s as number).toFixed(1)}s - ${(p.end_s as number).toFixed(1)}s]` : ""}${p.prompt_preview ? ` · ${(p.prompt_preview as string).slice(0, 70)}` : ""}`;
      case "subcall":
        return `${shortModelName(p.child_model)}${p.start_s != null && p.end_s != null ? ` · [${(p.start_s as number).toFixed(1)}s - ${(p.end_s as number).toFixed(1)}s]` : ""}${p.iterations_used != null ? ` · ${p.iterations_used} iters` : ""}${p.response_preview ? ` · ${(p.response_preview as string).slice(0, 70)}` : p.prompt_preview ? ` · ${(p.prompt_preview as string).slice(0, 70)}` : ""}`;
      case "sub_llm":
      case "vision":
      case "schema_generation":
        return `question=${p.question_chars ?? "?"}ch`;
      case "transcription":
        return `source=${p.source ?? "?"} path=${p.subtitle_path ?? "none"}`;
      default:
        return JSON.stringify(p ?? {}).slice(0, 80);
    }
  } catch {
    return event.kind;
  }
}

const CODE_KEYS = new Set([
  "code", "code_preview", "code_content",
  "response_content", "response_preview",
  "prompt_content", "prompt_preview",
  "stdout", "stderr",
  "final_answer",
  "feedback",
  "llm_queries_preview",
]);

const SKIP_KEYS = new Set([
  "prompt_chars", "response_chars", "code_chars",
  "question_chars", "n_frames",
]);

const IMAGE_PATH_KEYS = new Set([
  "frame_paths",
]);

function EventDetail({ payload }: { payload: Record<string, unknown> }) {
  const entries = Object.entries(payload).filter(
    ([k, v]) => !SKIP_KEYS.has(k) && v != null && v !== ""
  );

  if (entries.length === 0) {
    return (
      <span className="text-hud-dim italic">no payload data</span>
    );
  }

  return (
    <div className="space-y-1.5">
      {entries.map(([key, value]) => {
        if (IMAGE_PATH_KEYS.has(key) && Array.isArray(value) && value.length > 0) {
          return (
            <div key={key}>
              <span className="text-hud-dim uppercase tracking-wider">{key}: </span>
              <div className="mt-0.5 flex gap-1 overflow-x-auto py-1">
                {(value as string[]).map((path, i) => (
                  <Image
                    key={i}
                    src={frameUrl(path)}
                    alt={`frame ${i + 1}`}
                    width={160}
                    height={80}
                    unoptimized
                    className="h-20 w-auto shrink-0 border border-hud-border/50 object-cover"
                    loading="lazy"
                  />
                ))}
              </div>
            </div>
          );
        }

        const isCode = CODE_KEYS.has(key);
        const isLongString = typeof value === "string" && value.length > 100;
        const isObject = typeof value === "object" && !Array.isArray(value);
        const isArray = Array.isArray(value);

        return (
          <div key={key}>
            <span className="text-hud-dim uppercase tracking-wider">{key}: </span>
            {isCode || isLongString ? (
              <pre className="mt-0.5 max-h-48 overflow-auto border border-hud-border/50 bg-[#111] px-2 py-1 whitespace-pre-wrap break-all text-foreground/80">
                {String(value)}
              </pre>
            ) : isObject || isArray ? (
              <pre className="mt-0.5 max-h-48 overflow-auto border border-hud-border/50 bg-[#111] px-2 py-1 whitespace-pre-wrap break-all text-foreground/80">
                {JSON.stringify(value, null, 2)}
              </pre>
            ) : (
              <span className="text-foreground/80">
                {typeof value === "number"
                  ? Number.isInteger(value) ? value : value.toFixed(4)
                  : String(value)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

const VISIBLE_KINDS = new Set([
  "run_start",
  "run_end",
  "iteration_end",
  "root_response",
  "code_instruction",
  "code_execution",
  "subcall_start",
  "subcall",
  "sub_llm",
  "video_inspection",
  "frame_inspection",
  "audio_analysis",
  "tool_call",
  "schema_generation",
  "vision",
]);

export function TraceTimeline({ events, startTime }: TraceTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const visibleEvents = events.filter((e) => VISIBLE_KINDS.has(e.kind));
  const displayDepths = deriveDisplayDepths(visibleEvents);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visibleEvents.length]);

  return (
    <Panel title="TRACE TIMELINE" className="flex h-full flex-col">
      {visibleEvents.length === 0 ? (
        <span className="text-[12px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div
          ref={scrollRef}
          className="min-h-0 flex-1 space-y-0 overflow-y-auto text-[12px]"
        >
          {visibleEvents.map((event, i) => {
            const diff = startTime && event.timestamp ? event.timestamp - startTime : 0;
            const relativeS = Number.isFinite(diff) ? diff.toFixed(1) : "0.0";
            const isExpanded = expandedIndex === i;
            const hasPayload = event.payload && Object.keys(event.payload).length > 0;
            const depth = displayDepths[i] ?? 0;

            return (
              <div
                key={i}
                className="border-b border-hud-border/30 last:border-0"
              >
                <button
                  type="button"
                  onClick={() => hasPayload && setExpandedIndex(isExpanded ? null : i)}
                  className={`flex w-full items-center gap-2 py-1 text-left ${
                    hasPayload ? "cursor-pointer hover:bg-[#141414]" : "cursor-default"
                  } ${depth > 0 ? "bg-[#0d1117]" : ""}`}
                  style={{
                    paddingLeft: `${6 + depth * 20}px`,
                    boxShadow: depth > 0 ? "inset 2px 0 0 rgba(120, 180, 255, 0.22)" : undefined,
                  }}
                >
                  <span className="w-14 shrink-0 text-right tabular-nums leading-none text-hud-dim">
                    +{relativeS}s
                  </span>
                  <span
                    className={`w-11 shrink-0 border px-1 py-0.5 text-center font-mono text-[10px] uppercase tracking-[0.18em] leading-none ${depthBadgeClass(depth)}`}
                  >
                    {depthBadgeLabel(depth)}
                  </span>
                  <span
                    className={`h-1.5 w-1.5 shrink-0 ${getDotColor(event.kind)}`}
                  />
                  <span
                    className={`w-32 shrink-0 uppercase font-bold tracking-wider leading-none ${getEventColor(event.kind)}`}
                  >
                    {eventLabel(event, depth)}
                  </span>
                  <span className="min-w-0 flex-1 truncate leading-none text-hud-dim">
                    {getEventSummary(event)}
                  </span>
                  {hasPayload && (
                    <span className="w-4 shrink-0 text-center text-hud-dim">
                      {isExpanded ? "−" : "+"}
                    </span>
                  )}
                </button>

                {isExpanded && event.payload && (
                  <div
                    className="mb-1.5 mr-2 mt-0.5 border border-hud-border/50 bg-[#0d0d0d] p-2"
                    style={{ marginLeft: `${118 + depth * 20}px` }}
                  >
                    <EventDetail payload={event.payload} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
