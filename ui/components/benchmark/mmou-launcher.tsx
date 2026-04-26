"use client";

import type { ChangeEvent, FormEvent, ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createMMOUJob,
  fetchMMOUCatalog,
  fetchMMOUJobs,
  fetchMMOUQuestionTrace,
  resumeMMOUJob,
  stopMMOUJob,
  streamMMOUJobEvents,
} from "@/lib/api";
import type { MMOUCatalog, MMOUJobSummary, TraceEvent } from "@/lib/types";
import { TraceTimeline } from "@/components/hud/trace-timeline";

interface MMOUFormState {
  limit: number;
  stratified: boolean;
  workers: number;
  maxIterations: number;
  maxDepth: number;
  maxBudgetUsd: string;
  maxTimeoutS: string;
  runName: string;
  outputDir: string;
  keepArtifacts: boolean;
}

function formatTime(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}

function formatTimestamp(ts: number | null): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function traceEventKey(event: TraceEvent): string {
  return `${event.kind}:${event.timestamp}:${JSON.stringify(event.payload)}`;
}

function shortModel(model: string | null | undefined): string {
  if (!model) return "—";
  const parts = model.split("/");
  return parts[parts.length - 1];
}

function statusTone(status: MMOUJobSummary["status"] | "active") {
  switch (status) {
    case "running":
    case "active":
      return "text-hud-green border-hud-green/40 bg-hud-green/10";
    case "stopping":
      return "text-hud-amber border-hud-amber/40 bg-hud-amber/10";
    case "complete":
      return "text-hud-cyan border-hud-cyan/40 bg-hud-cyan/10";
    case "stopped":
    case "interrupted":
      return "text-hud-dim border-hud-border bg-black/20";
    case "error":
      return "text-hud-red border-hud-red/40 bg-hud-red/10";
    default:
      return "text-hud-amber border-hud-amber/40 bg-hud-amber/10";
  }
}

function isActiveJob(job: MMOUJobSummary): boolean {
  return job.status === "pending" || job.status === "running" || job.status === "stopping";
}

function chooseQuestionSelection(job: MMOUJobSummary, currentQuestionId: string | null): string | null {
  if (currentQuestionId && job.question_ids.includes(currentQuestionId)) return currentQuestionId;
  if (job.active_question_ids.length > 0) return job.active_question_ids[0];
  const started = job.questions.find((question) => question.status !== "pending");
  if (started) return started.question_id;
  return job.question_ids[0] ?? null;
}

function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <span className="mb-1 block text-[11px] font-bold uppercase tracking-[0.14em] text-hud-dim">
      {children}
    </span>
  );
}

function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step ?? 1}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full border border-hud-border bg-black/30 px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-hud-green disabled:cursor-not-allowed disabled:opacity-50"
    />
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full border border-hud-border bg-black/30 px-3 py-2 text-sm text-foreground outline-none transition-colors placeholder:text-hud-dim/60 focus:border-hud-green"
    />
  );
}

function CheckboxRow({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint: string;
  disabled?: boolean;
}) {
  return (
    <label className="flex items-start gap-3 border border-hud-border bg-black/20 px-3 py-2 text-sm">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1"
      />
      <span>
        <span className="block text-foreground">{label}</span>
        <span className="block text-[12px] text-hud-dim">{hint}</span>
      </span>
    </label>
  );
}

function extractQuestionIds(payload: unknown): string[] {
  const rawItems = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object" && Array.isArray((payload as { question_ids?: unknown }).question_ids)
      ? (payload as { question_ids: unknown[] }).question_ids
      : null;
  if (!rawItems) {
    throw new Error("JSON must be an array or an object with question_ids.");
  }

  const seen = new Set<string>();
  const ids: string[] = [];
  for (const item of rawItems) {
    const value = typeof item === "string"
      ? item
      : item && typeof item === "object" && "question_id" in item
        ? String((item as { question_id?: unknown }).question_id ?? "")
        : "";
    const questionId = value.trim();
    if (!questionId || seen.has(questionId)) continue;
    seen.add(questionId);
    ids.push(questionId);
  }
  if (ids.length === 0) {
    throw new Error("No question_ids found in JSON.");
  }
  return ids;
}

function MMOUJobCard({
  job,
  expanded,
  onToggleExpanded,
  onExpand,
  onSnapshot,
}: {
  job: MMOUJobSummary;
  expanded: boolean;
  onToggleExpanded: () => void;
  onExpand: () => void;
  onSnapshot: (job: MMOUJobSummary) => void;
}) {
  const [manualQuestionId, setManualQuestionId] = useState<string | null>(null);
  const [traceByQuestion, setTraceByQuestion] = useState<Record<string, TraceEvent[]>>({});
  const [traceError, setTraceError] = useState<string | null>(null);
  const [stopPending, setStopPending] = useState(false);
  const [resumePending, setResumePending] = useState(false);
  const seenKeysRef = useRef<Record<string, Set<string>>>({});
  const onSnapshotRef = useRef(onSnapshot);
  const selectedQuestionId = useMemo(
    () => chooseQuestionSelection(job, manualQuestionId),
    [job, manualQuestionId]
  );

  useEffect(() => {
    onSnapshotRef.current = onSnapshot;
  }, [onSnapshot]);

  useEffect(() => {
    if (!expanded || !selectedQuestionId) return;
    let cancelled = false;
    fetchMMOUQuestionTrace(job.job_id, selectedQuestionId)
      .then((trace) => {
        if (cancelled) return;
        seenKeysRef.current[selectedQuestionId] = new Set(trace.events.map(traceEventKey));
        setTraceByQuestion((current) => ({ ...current, [selectedQuestionId]: trace.events }));
      })
      .catch((error: Error) => {
        if (!cancelled) setTraceError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [expanded, job.job_id, selectedQuestionId]);

  useEffect(() => {
    if (!expanded || (job.status !== "pending" && job.status !== "running" && job.status !== "stopping")) return;
    const stop = streamMMOUJobEvents(
      job.job_id,
      (snapshot) => {
        onSnapshotRef.current(snapshot);
        if (snapshot.status !== "stopping") {
          setStopPending(false);
        }
      },
      (questionId, event) => {
        setTraceByQuestion((current) => {
          const seen = seenKeysRef.current[questionId] ?? new Set<string>();
          seenKeysRef.current[questionId] = seen;
          const key = traceEventKey(event);
          if (seen.has(key)) return current;
          seen.add(key);
          return { ...current, [questionId]: [...(current[questionId] ?? []), event] };
        });
      },
      (error) => setTraceError(error),
      () => undefined
    );
    return stop;
  }, [expanded, job.job_id, job.status]);

  const selectedQuestion = job.questions.find((question) => question.question_id === selectedQuestionId) ?? null;
  const selectedTrace = selectedQuestionId ? traceByQuestion[selectedQuestionId] ?? [] : [];
  const progress = job.total_questions > 0
    ? (job.completed_questions + job.error_questions) / job.total_questions
    : 0;
  const isStoppable = job.status === "pending" || job.status === "running" || job.status === "stopping";
  const isResumable = !isStoppable && (job.status === "interrupted" || job.status === "stopped" || job.status === "error" || job.error_questions > 0);
  const resumeActionLabel = job.error_questions > 0 || job.status === "error" ? "Retry" : "Resume";
  const resumePendingLabel = resumeActionLabel === "Retry" ? "Retrying…" : "Resuming…";

  const handleStop = async () => {
    if (!isStoppable || stopPending || job.status === "stopping") return;
    setStopPending(true);
    try {
      const snapshot = await stopMMOUJob(job.job_id);
      onSnapshotRef.current(snapshot);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to stop MMOU job";
      setTraceError(message);
    } finally {
      setStopPending(false);
    }
  };

  const handleResume = async () => {
    if (!isResumable || resumePending) return;
    onExpand();
    setResumePending(true);
    try {
      const snapshot = await resumeMMOUJob(job.job_id);
      onSnapshotRef.current(snapshot);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to resume MMOU job";
      setTraceError(message);
    } finally {
      setResumePending(false);
    }
  };

  return (
    <div className="min-w-0 border border-hud-border bg-hud-panel">
      <div
        className={`border-b border-hud-border px-4 py-3 outline-none transition-colors hover:bg-white/[0.02] focus:border-hud-green ${
          expanded ? "" : "border-b-0"
        }`}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <button type="button" onClick={onToggleExpanded} className="min-w-0 text-left">
            <div className="flex flex-wrap items-center gap-2">
              <span className="w-4 shrink-0 text-[12px] text-hud-dim">{expanded ? "-" : "+"}</span>
              <span className="min-w-0 break-words text-sm font-bold text-foreground">{job.run_name}</span>
              <span className={`border px-2 py-0.5 text-[11px] font-bold uppercase tracking-[0.14em] ${statusTone(job.status)}`}>
                {job.status}
              </span>
              <span className="border border-hud-border bg-black/20 px-2 py-0.5 text-[11px] uppercase tracking-[0.14em] text-hud-dim">
                {job.selection_source === "question_ids" ? "question ids" : job.stratified ? "stratified" : "dataset order"}
              </span>
            </div>
            <p className="mt-1 text-[12px] text-hud-dim">
              Created {formatTimestamp(job.created_at)} · {job.total_questions} questions · workers {job.workers}
            </p>
          </button>
          <div className="grid grid-cols-2 gap-3 text-right text-[12px] sm:grid-cols-3">
            <div>
              <span className="block uppercase tracking-[0.14em] text-hud-dim">Progress</span>
              <span className="font-bold text-foreground">{job.completed_questions + job.error_questions}/{job.total_questions}</span>
            </div>
            <div>
              <span className="block uppercase tracking-[0.14em] text-hud-dim">Errors</span>
              <span className={job.error_questions > 0 ? "font-bold text-hud-red" : "font-bold text-foreground"}>{job.error_questions}</span>
            </div>
            <div>
              <span className="block uppercase tracking-[0.14em] text-hud-dim">Budget</span>
              <span className="font-bold text-hud-amber">{job.max_budget_usd != null ? `$${job.max_budget_usd.toFixed(2)}` : "none"}</span>
            </div>
          </div>
        </div>
        {(job.status === "stopping" || job.status === "stopped" || job.status === "interrupted") && (
          <p className="mt-2 text-[12px] text-hud-amber">
            {job.status === "stopping"
              ? "Stop requested. No new MMOU questions will start; already-running questions are draining."
              : job.status === "interrupted"
                ? "Run was interrupted while the API was offline. Resume will skip completed predictions."
                : "Run stopped. Pending questions were not started."}
          </p>
        )}
        <div className="mt-3 flex justify-end gap-2">
          {isResumable && (
            <button
              type="button"
              onClick={handleResume}
              disabled={resumePending}
              className="border border-hud-green/50 px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.14em] text-hud-green transition-colors hover:bg-hud-green/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {resumePending ? resumePendingLabel : resumeActionLabel}
            </button>
          )}
          {isStoppable && (
            <button
              type="button"
              onClick={handleStop}
              disabled={stopPending || job.status === "stopping"}
              className="border border-hud-red/50 px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.14em] text-hud-red transition-colors hover:bg-hud-red/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {job.status === "stopping" || stopPending ? "Stopping…" : "Stop"}
            </button>
          )}
          <button
            type="button"
            onClick={onToggleExpanded}
            className="border border-hud-border px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.14em] text-hud-dim transition-colors hover:text-foreground"
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
        </div>
        <div className="mt-3 h-2 overflow-hidden border border-hud-border bg-black/30">
          <div
            className="h-full bg-hud-green transition-[width]"
            style={{ width: `${Math.max(progress * 100, job.status === "complete" ? 100 : 0)}%` }}
          />
        </div>
      </div>

      {expanded && (
      <div className="grid gap-4 p-4 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <div>
            <FieldLabel>Question Queue</FieldLabel>
            <div className="grid max-h-[520px] grid-cols-1 gap-2 overflow-y-auto pr-1">
              {job.questions.map((question) => {
                const selected = question.question_id === selectedQuestionId;
                return (
                  <button
                    key={question.question_id}
                    type="button"
                    onClick={() => setManualQuestionId(question.question_id)}
                    className={`border px-3 py-2 text-left text-[12px] transition-colors ${
                      selected
                        ? "border-hud-green bg-hud-green/10 text-foreground"
                        : "border-hud-border bg-black/20 text-hud-dim hover:text-foreground"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="min-w-0 truncate font-bold">{question.question_id}</span>
                      <span className={`border px-1.5 py-0.5 uppercase tracking-[0.14em] ${statusTone(question.status === "running" ? "active" : question.status)}`}>
                        {question.status}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center justify-between gap-2">
                      <span className="truncate text-foreground/80">{question.domain}</span>
                      <span className="shrink-0 text-hud-dim">{question.answer ?? "—"}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="border border-hud-border bg-black/20 p-3 text-[12px]">
            <FieldLabel>Job Config</FieldLabel>
            <div className="space-y-1 text-hud-dim">
              <div>Iterations: <span className="text-foreground">{job.max_iterations}</span></div>
              <div>Max depth: <span className="text-foreground">{job.max_depth}</span></div>
              <div>Domains: <span className="text-foreground">{job.domains?.join(", ") || "all"}</span></div>
              <div>Keep artifacts: <span className="text-foreground">{job.keep_artifacts ? "on" : "off"}</span></div>
              <div>Root model: <span className="text-foreground">{shortModel(job.models.root)}</span></div>
              <div>Recursive model: <span className="text-foreground">{shortModel(job.models.recursive)}</span></div>
              <div>Sub model: <span className="text-foreground">{shortModel(job.models.sub)}</span></div>
              <div>Job dir: <span className="break-all text-foreground">{job.job_dir}</span></div>
            </div>
          </div>

          {(job.stderr_tail.length > 0 || traceError) && (
            <div className="border border-hud-red/40 bg-hud-red/5 p-3 text-[12px]">
              <FieldLabel>Errors</FieldLabel>
              <div className="space-y-1 text-hud-red">
                {traceError && <div>{traceError}</div>}
                {job.stderr_tail.slice(-4).map((line, index) => (
                  <div key={`${line}-${index}`} className="break-words">{line}</div>
                ))}
              </div>
            </div>
          )}

          {job.stdout_tail.length > 0 && (
            <div className="border border-hud-border bg-black/20 p-3 text-[12px]">
              <FieldLabel>Recent Logs</FieldLabel>
              <div className="space-y-1 text-hud-dim">
                {job.stdout_tail.slice(-6).map((line, index) => (
                  <div key={`${line}-${index}`} className="break-words">{line}</div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="min-w-0 space-y-4">
          {selectedQuestion ? (
            <>
              <div className="min-w-0 border border-hud-border bg-black/20 p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="min-w-0 break-words text-sm font-bold text-foreground">
                        {selectedQuestion.question_id}
                      </span>
                      <span className={`border px-2 py-0.5 text-[11px] uppercase tracking-[0.14em] ${statusTone(selectedQuestion.status === "running" ? "active" : selectedQuestion.status)}`}>
                        {selectedQuestion.status}
                      </span>
                      <span className="border border-hud-border bg-black/20 px-2 py-0.5 text-[11px] text-hud-dim">
                        {selectedQuestion.domain} / {selectedQuestion.subdomain || "—"}
                      </span>
                    </div>
                    <p className="mt-2 break-words text-[12px] text-hud-dim">{selectedQuestion.question}</p>
                    <div className="mt-2 grid gap-1 text-[12px] text-hud-dim">
                      {Object.entries(selectedQuestion.options).map(([letter, option]) => (
                        <div key={letter} className={selectedQuestion.answer === letter ? "text-hud-green" : ""}>
                          <span className="font-bold">{letter}.</span> {option}
                        </div>
                      ))}
                    </div>
                    {(selectedQuestion.error || selectedQuestion.parse_error) && (
                      <p className="mt-2 break-words text-[12px] text-hud-red">
                        {selectedQuestion.error || selectedQuestion.parse_error}
                      </p>
                    )}
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-right text-[12px]">
                    <div>
                      <span className="block uppercase tracking-[0.14em] text-hud-dim">Answer</span>
                      <span className="font-bold text-foreground">{selectedQuestion.answer ?? "—"}</span>
                    </div>
                    <div>
                      <span className="block uppercase tracking-[0.14em] text-hud-dim">Cost</span>
                      <span className="font-bold text-hud-amber">
                        {selectedQuestion.cost_usd != null ? `$${selectedQuestion.cost_usd.toFixed(4)}` : "—"}
                      </span>
                    </div>
                    <div>
                      <span className="block uppercase tracking-[0.14em] text-hud-dim">Time</span>
                      <span className="font-bold text-foreground">{formatTime(selectedQuestion.wall_time_s)}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="min-w-0 overflow-hidden">
                <TraceTimeline
                  events={selectedTrace}
                  startTime={selectedTrace[0]?.timestamp ?? selectedQuestion.started_at ?? null}
                />
              </div>
            </>
          ) : (
            <div className="border border-hud-border bg-black/20 px-4 py-8 text-center text-[12px] uppercase tracking-[0.14em] text-hud-dim">
              No MMOU question selected
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  );
}

export function MMOULauncher() {
  const [catalog, setCatalog] = useState<MMOUCatalog | null>(null);
  const [jobs, setJobs] = useState<MMOUJobSummary[]>([]);
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [uploadedQuestionIds, setUploadedQuestionIds] = useState<string[]>([]);
  const [uploadLabel, setUploadLabel] = useState<string | null>(null);
  const [expandedJobIds, setExpandedJobIds] = useState<Set<string>>(new Set());
  const [form, setForm] = useState<MMOUFormState>({
    limit: 10,
    stratified: true,
    workers: 1,
    maxIterations: 8,
    maxDepth: 2,
    maxBudgetUsd: "",
    maxTimeoutS: "",
    runName: "",
    outputDir: "",
    keepArtifacts: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMMOUCatalog()
      .then((nextCatalog) => {
        setCatalog(nextCatalog);
        setForm({
          limit: nextCatalog.defaults.limit,
          stratified: nextCatalog.defaults.stratified,
          workers: nextCatalog.defaults.workers,
          maxIterations: nextCatalog.defaults.max_iterations,
          maxDepth: nextCatalog.defaults.max_depth,
          maxBudgetUsd: nextCatalog.defaults.max_budget_usd == null ? "" : String(nextCatalog.defaults.max_budget_usd),
          maxTimeoutS: nextCatalog.defaults.max_timeout_s == null ? "" : String(nextCatalog.defaults.max_timeout_s),
          runName: "",
          outputDir: nextCatalog.defaults.output_dir,
          keepArtifacts: nextCatalog.defaults.keep_artifacts,
        });
      })
      .catch((nextError: Error) => setError(nextError.message));

    fetchMMOUJobs()
      .then((nextJobs) => {
        setJobs(nextJobs);
        setExpandedJobIds(new Set(nextJobs.filter(isActiveJob).map((job) => job.job_id)));
      })
      .catch((nextError: Error) => setError(nextError.message));
  }, []);

  const domainEntries = useMemo(() => (
    Object.entries(catalog?.domain_counts ?? {}).sort(([left], [right]) => left.localeCompare(right))
  ), [catalog]);
  const defaultModels = catalog?.defaults.models ?? null;
  const explicitQuestionSelection = uploadedQuestionIds.length > 0;

  const updateJob = useCallback((nextJob: MMOUJobSummary) => {
    setJobs((current) => {
      const existing = current.some((job) => job.job_id === nextJob.job_id);
      const next = existing
        ? current.map((job) => (job.job_id === nextJob.job_id ? nextJob : job))
        : [nextJob, ...current];
      return [...next].sort((a, b) => b.created_at - a.created_at);
    });
    if (isActiveJob(nextJob)) {
      setExpandedJobIds((current) => new Set([...current, nextJob.job_id]));
    }
  }, []);

  const expandJob = useCallback((jobId: string) => {
    setExpandedJobIds((current) => new Set([...current, jobId]));
  }, []);

  const toggleJobExpanded = useCallback((jobId: string) => {
    setExpandedJobIds((current) => {
      const next = new Set(current);
      if (next.has(jobId)) {
        next.delete(jobId);
      } else {
        next.add(jobId);
      }
      return next;
    });
  }, []);

  const toggleDomain = (domain: string) => {
    setSelectedDomains((current) => (
      current.includes(domain)
        ? current.filter((item) => item !== domain)
        : [...current, domain].sort((a, b) => a.localeCompare(b))
    ));
  };

  const handleQuestionFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const ids = extractQuestionIds(parsed);
      setUploadedQuestionIds(ids);
      setUploadLabel(`${file.name} · ${ids.length} ids`);
      setForm((current) => ({ ...current, limit: Math.min(current.limit, ids.length) }));
      setError(null);
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : "Could not parse question ID JSON";
      setError(message);
    }
  };

  const optionalNumber = (value: string): number | null => {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new Error("Optional numeric fields must be positive.");
    }
    return parsed;
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const job = await createMMOUJob({
        benchmark_type: "mmou",
        limit: form.limit,
        stratified: form.stratified,
        domains: selectedDomains.length > 0 ? selectedDomains : null,
        question_ids: explicitQuestionSelection ? uploadedQuestionIds : null,
        workers: form.workers,
        max_iterations: form.maxIterations,
        max_depth: form.maxDepth,
        max_budget_usd: optionalNumber(form.maxBudgetUsd),
        max_timeout_s: optionalNumber(form.maxTimeoutS),
        output_dir: form.outputDir.trim() || null,
        run_name: form.runName.trim() || null,
        keep_artifacts: form.keepArtifacts,
      });
      updateJob(job);
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Failed to create MMOU job";
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="min-w-0 border border-hud-border bg-hud-panel">
      <div className="border-b border-hud-border px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-[0.18em] text-hud-label">Run MMOU</h2>
            <p className="mt-1 text-[12px] text-hud-dim">
              Launch MMOU RLM batches, stream per-question traces, and resume persisted runs.
            </p>
          </div>
          <div className="text-right text-[12px] text-hud-dim">
            <div>{jobs.filter((job) => job.status === "running" || job.status === "pending" || job.status === "stopping").length} active jobs</div>
            <div>{catalog ? `${catalog.total_questions} catalog questions` : "catalog loading"}</div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 p-4 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <form onSubmit={handleSubmit} className="min-w-0 space-y-4">
          <div className="border border-hud-border bg-black/20 p-3">
            <FieldLabel>Question Selection</FieldLabel>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <FieldLabel>Total Tasks</FieldLabel>
                <NumberInput
                  value={form.limit}
                  min={1}
                  max={explicitQuestionSelection ? uploadedQuestionIds.length : undefined}
                  onChange={(value) => setForm((current) => ({
                    ...current,
                    limit: explicitQuestionSelection ? Math.min(value, uploadedQuestionIds.length) : value,
                  }))}
                />
              </div>
              <div className="flex items-end">
                <label className="flex h-[38px] w-full items-center gap-3 border border-hud-border bg-black/30 px-3 text-sm text-foreground">
                  <input
                    type="checkbox"
                    checked={form.stratified}
                    disabled={explicitQuestionSelection}
                    onChange={(e) => setForm((current) => ({ ...current, stratified: e.target.checked }))}
                  />
                  <span>Stratify domains</span>
                </label>
              </div>
            </div>
            <p className="mt-2 text-[12px] text-hud-dim">
              {explicitQuestionSelection
                ? `Running the first ${Math.min(form.limit, uploadedQuestionIds.length)} of ${uploadedQuestionIds.length} uploaded question IDs. Stratification and domains are ignored.`
                : form.stratified
                  ? "Balanced round-robin selection across selected domains."
                  : "Dataset order selection from selected domains."}
            </p>
          </div>

          <div className="border border-hud-border bg-black/20 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <FieldLabel>Question ID JSON</FieldLabel>
                <p className="text-[12px] text-hud-dim">
                  Accepts arrays, question_ids objects, or rows with question_id.
                </p>
              </div>
              {uploadedQuestionIds.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setUploadedQuestionIds([]);
                    setUploadLabel(null);
                  }}
                  className="border border-hud-border px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.14em] text-hud-dim transition-colors hover:text-foreground"
                >
                  Clear
                </button>
              )}
            </div>
            <input
              type="file"
              accept="application/json,.json"
              onChange={handleQuestionFile}
              className="mt-3 block w-full text-[12px] text-hud-dim file:mr-3 file:border file:border-hud-border file:bg-black/30 file:px-3 file:py-1.5 file:text-[12px] file:font-bold file:uppercase file:tracking-[0.14em] file:text-foreground"
            />
            {uploadLabel && <p className="mt-2 text-[12px] text-hud-green">{uploadLabel}</p>}
          </div>

          <div className="border border-hud-border bg-black/20 p-3">
            <div className="flex items-center justify-between gap-3">
              <FieldLabel>Domains</FieldLabel>
              <button
                type="button"
                disabled={explicitQuestionSelection}
                onClick={() => setSelectedDomains([])}
                className="text-[12px] text-hud-dim transition-colors hover:text-foreground disabled:opacity-50"
              >
                All domains
              </button>
            </div>
            <div className="mt-2 grid max-h-56 grid-cols-1 gap-2 overflow-y-auto pr-1">
              {domainEntries.map(([domain, count]) => {
                const checked = selectedDomains.includes(domain);
                return (
                  <label
                    key={domain}
                    className={`flex items-center justify-between gap-3 border px-3 py-2 text-sm transition-colors ${
                      checked
                        ? "border-hud-green bg-hud-green/10"
                        : "border-hud-border bg-black/10"
                    } ${explicitQuestionSelection ? "opacity-50" : ""}`}
                  >
                    <span className="flex min-w-0 items-center gap-3">
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={explicitQuestionSelection}
                        onChange={() => toggleDomain(domain)}
                      />
                      <span className="min-w-0 truncate text-foreground">{domain}</span>
                    </span>
                    <span className="shrink-0 text-[12px] text-hud-dim">{count}</span>
                  </label>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <FieldLabel>Workers</FieldLabel>
              <NumberInput value={form.workers} min={1} onChange={(value) => setForm((current) => ({ ...current, workers: value }))} />
            </div>
            <div>
              <FieldLabel>Max Iterations</FieldLabel>
              <NumberInput value={form.maxIterations} min={1} onChange={(value) => setForm((current) => ({ ...current, maxIterations: value }))} />
            </div>
            <div>
              <FieldLabel>Max Depth</FieldLabel>
              <NumberInput value={form.maxDepth} min={1} onChange={(value) => setForm((current) => ({ ...current, maxDepth: value }))} />
            </div>
            <div>
              <FieldLabel>Budget / Question</FieldLabel>
              <TextInput value={form.maxBudgetUsd} onChange={(value) => setForm((current) => ({ ...current, maxBudgetUsd: value }))} placeholder="none" />
            </div>
            <div>
              <FieldLabel>Timeout / Question</FieldLabel>
              <TextInput value={form.maxTimeoutS} onChange={(value) => setForm((current) => ({ ...current, maxTimeoutS: value }))} placeholder="none" />
            </div>
            <div>
              <FieldLabel>Run Name</FieldLabel>
              <TextInput value={form.runName} onChange={(value) => setForm((current) => ({ ...current, runName: value }))} placeholder="optional" />
            </div>
          </div>

          <div>
            <FieldLabel>Output Dir</FieldLabel>
            <TextInput value={form.outputDir} onChange={(value) => setForm((current) => ({ ...current, outputDir: value }))} />
          </div>

          {defaultModels && (
            <div className="border border-hud-border bg-black/20 p-3 text-[12px]">
              <FieldLabel>Model Config</FieldLabel>
              <div className="space-y-1 text-hud-dim">
                <div>Root model: <span className="text-foreground">{shortModel(defaultModels.root)}</span></div>
                <div>Recursive model: <span className="text-foreground">{shortModel(defaultModels.recursive)}</span></div>
                <div>Sub model: <span className="text-foreground">{shortModel(defaultModels.sub)}</span></div>
                <div>Vision model: <span className="text-foreground">{shortModel(defaultModels.vision)}</span></div>
              </div>
            </div>
          )}

          <CheckboxRow
            checked={form.keepArtifacts}
            onChange={(checked) => setForm((current) => ({ ...current, keepArtifacts: checked }))}
            label="Keep full per-question artifacts"
            hint="Off by default to avoid retaining downloaded videos and intermediate media files."
          />

          {error && (
            <div className="border border-hud-red/40 bg-hud-red/5 px-3 py-2 text-sm text-hud-red">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !catalog}
            className="w-full border border-foreground px-4 py-2 text-[12px] font-bold uppercase tracking-[0.14em] text-foreground transition-colors hover:bg-foreground hover:text-background disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "Starting…" : "Start MMOU Run"}
          </button>
        </form>

        <div className="min-w-0 space-y-4">
          {jobs.length === 0 ? (
            <div className="border border-hud-border bg-black/20 px-4 py-10 text-center text-[12px] uppercase tracking-[0.14em] text-hud-dim">
              No MMOU jobs yet
            </div>
          ) : (
            jobs.map((job) => (
              <MMOUJobCard
                key={job.job_id}
                job={job}
                expanded={expandedJobIds.has(job.job_id)}
                onToggleExpanded={() => toggleJobExpanded(job.job_id)}
                onExpand={() => expandJob(job.job_id)}
                onSnapshot={updateJob}
              />
            ))
          )}
        </div>
      </div>
    </section>
  );
}
