"use client";

import type { FormEvent, ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createBenchmarkJob,
  fetchBenchmarkCatalog,
  fetchBenchmarkJobs,
  fetchBenchmarkPromptTrace,
  stopBenchmarkJob,
  streamBenchmarkJobEvents,
} from "@/lib/api";
import type {
  BenchmarkCatalog,
  BenchmarkJobSummary,
  BenchmarkPromptCatalogItem,
  TraceEvent,
} from "@/lib/types";
import { TraceTimeline } from "@/components/hud/trace-timeline";

type PromptPreset = "all" | "demo" | "lvb" | "custom";

interface LauncherFormState {
  workers: number;
  maxIterations: number;
  maxDepth: number;
  maxBudgetUsd: number;
  fast: boolean;
  runName: string;
  outputDir: string;
  downloadLvb: boolean;
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

function promptKey(event: TraceEvent): string {
  return `${event.kind}:${event.timestamp}:${JSON.stringify(event.payload)}`;
}

function shortModel(model: string | null | undefined): string {
  if (!model) return "—";
  const parts = model.split("/");
  return parts[parts.length - 1];
}

function statusTone(status: BenchmarkJobSummary["status"] | "active") {
  switch (status) {
    case "running":
    case "active":
      return "text-hud-green border-hud-green/40 bg-hud-green/10";
    case "stopping":
      return "text-hud-amber border-hud-amber/40 bg-hud-amber/10";
    case "complete":
      return "text-hud-cyan border-hud-cyan/40 bg-hud-cyan/10";
    case "stopped":
      return "text-hud-dim border-hud-border bg-black/20";
    case "error":
      return "text-hud-red border-hud-red/40 bg-hud-red/10";
    default:
      return "text-hud-amber border-hud-amber/40 bg-hud-amber/10";
  }
}

function choosePromptSelection(job: BenchmarkJobSummary, currentPromptId: number | null): number | null {
  if (currentPromptId && job.prompt_ids.includes(currentPromptId)) return currentPromptId;
  if (job.active_prompt_ids.length > 0) return job.active_prompt_ids[0];
  const started = job.prompts.find((prompt) => prompt.status !== "pending");
  if (started) return started.prompt_id;
  return job.prompt_ids[0] ?? null;
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
  step,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  step?: number;
}) {
  return (
    <input
      type="number"
      min={min}
      step={step ?? 1}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full border border-hud-border bg-black/30 px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-hud-green"
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
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint: string;
}) {
  return (
    <label className="flex items-start gap-3 border border-hud-border bg-black/20 px-3 py-2 text-sm">
      <input
        type="checkbox"
        checked={checked}
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

function BenchmarkJobCard({
  job,
  onSnapshot,
  onSettled,
}: {
  job: BenchmarkJobSummary;
  onSnapshot: (job: BenchmarkJobSummary) => void;
  onSettled: () => void;
}) {
  const [manualPromptId, setManualPromptId] = useState<number | null>(null);
  const [traceByPrompt, setTraceByPrompt] = useState<Record<number, TraceEvent[]>>({});
  const [traceError, setTraceError] = useState<string | null>(null);
  const [stopPending, setStopPending] = useState(false);
  const seenKeysRef = useRef<Record<number, Set<string>>>({});
  const previousStatusRef = useRef(job.status);
  const onSnapshotRef = useRef(onSnapshot);
  const onSettledRef = useRef(onSettled);
  const selectedPromptId = useMemo(
    () => choosePromptSelection(job, manualPromptId),
    [job, manualPromptId]
  );

  useEffect(() => {
    onSnapshotRef.current = onSnapshot;
  }, [onSnapshot]);

  useEffect(() => {
    onSettledRef.current = onSettled;
  }, [onSettled]);

  useEffect(() => {
    if (!selectedPromptId) return;
    let cancelled = false;
    fetchBenchmarkPromptTrace(job.job_id, selectedPromptId)
      .then((trace) => {
        if (cancelled) return;
        seenKeysRef.current[selectedPromptId] = new Set(trace.events.map(promptKey));
        setTraceByPrompt((current) => ({ ...current, [selectedPromptId]: trace.events }));
      })
      .catch((error: Error) => {
        if (!cancelled) setTraceError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [job.job_id, selectedPromptId]);

  useEffect(() => {
    if (job.status !== "pending" && job.status !== "running" && job.status !== "stopping") return;
    const stop = streamBenchmarkJobEvents(
      job.job_id,
      (snapshot) => {
        const previousStatus = previousStatusRef.current;
        previousStatusRef.current = snapshot.status;
        onSnapshotRef.current(snapshot);
        if (
          (previousStatus === "pending" || previousStatus === "running" || previousStatus === "stopping") &&
          (snapshot.status === "complete" || snapshot.status === "error" || snapshot.status === "stopped")
        ) {
          onSettledRef.current();
        }
        if (snapshot.status !== "stopping") {
          setStopPending(false);
        }
      },
      (promptId, event) => {
        setTraceByPrompt((current) => {
          const seen = seenKeysRef.current[promptId] ?? new Set<string>();
          seenKeysRef.current[promptId] = seen;
          const key = promptKey(event);
          if (seen.has(key)) return current;
          seen.add(key);
          const nextEvents = [...(current[promptId] ?? []), event];
          return { ...current, [promptId]: nextEvents };
        });
      },
      (error) => setTraceError(error),
      () => undefined
    );
    return stop;
  }, [job.job_id, job.status]);

  const selectedPrompt = job.prompts.find((prompt) => prompt.prompt_id === selectedPromptId) ?? null;
  const selectedTrace = selectedPromptId ? traceByPrompt[selectedPromptId] ?? [] : [];
  const progress = job.total_prompts > 0 ? (job.completed_prompts + job.error_prompts) / job.total_prompts : 0;
  const isStoppable = job.status === "pending" || job.status === "running" || job.status === "stopping";

  const handleStop = async () => {
    if (!isStoppable || stopPending || job.status === "stopping") return;
    setStopPending(true);
    onSnapshotRef.current({
      ...job,
      status: "stopping",
      stop_requested_at: Date.now() / 1000,
      stop_reason: "Stop requested from dashboard",
    });
    try {
      const snapshot = await stopBenchmarkJob(job.job_id);
      onSnapshotRef.current(snapshot);
      if (snapshot.status !== "stopping") {
        setStopPending(false);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to stop benchmark job";
      setTraceError(message);
      setStopPending(false);
    }
  };

  return (
    <div className="min-w-0 border border-hud-border bg-hud-panel">
      <div className="border-b border-hud-border px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="min-w-0 break-words text-sm font-bold text-foreground">{job.run_name}</span>
              <span className={`border px-2 py-0.5 text-[11px] font-bold uppercase tracking-[0.14em] ${statusTone(job.status)}`}>
                {job.status}
              </span>
            </div>
            <p className="mt-1 text-[12px] text-hud-dim">
              Created {formatTimestamp(job.created_at)} · {job.total_prompts} prompts · workers {job.workers}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 text-right text-[12px] sm:grid-cols-3">
            <div>
              <span className="block uppercase tracking-[0.14em] text-hud-dim">Progress</span>
              <span className="font-bold text-foreground">{job.completed_prompts + job.error_prompts}/{job.total_prompts}</span>
            </div>
            <div>
              <span className="block uppercase tracking-[0.14em] text-hud-dim">Errors</span>
              <span className={job.error_prompts > 0 ? "font-bold text-hud-red" : "font-bold text-foreground"}>{job.error_prompts}</span>
            </div>
            <div>
              <span className="block uppercase tracking-[0.14em] text-hud-dim">Budget</span>
              <span className="font-bold text-hud-amber">${job.max_budget_usd.toFixed(2)}</span>
            </div>
          </div>
        </div>
        {(job.status === "stopping" || job.status === "stopped") && (
          <p className="mt-2 text-[12px] text-hud-amber">
            {job.status === "stopping"
              ? "Stop requested. No new prompts will start; already-running prompts are draining."
              : "Run stopped. Pending prompts were not started."}
          </p>
        )}
        {isStoppable && (
          <div className="mt-3 flex justify-end">
            <button
              type="button"
              onClick={handleStop}
              disabled={stopPending || job.status === "stopping"}
              className="border border-hud-red/50 px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.14em] text-hud-red transition-colors hover:bg-hud-red/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {job.status === "stopping" || stopPending ? "Stopping…" : "Stop"}
            </button>
          </div>
        )}
        <div className="mt-3 h-2 overflow-hidden border border-hud-border bg-black/30">
          <div
            className="h-full bg-hud-green transition-[width]"
            style={{ width: `${Math.max(progress * 100, job.status === "complete" ? 100 : 0)}%` }}
          />
        </div>
      </div>

      <div className="grid gap-4 p-4 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <div>
            <FieldLabel>Prompt Queue</FieldLabel>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {job.prompts.map((prompt) => {
                const selected = prompt.prompt_id === selectedPromptId;
                return (
                  <button
                    key={prompt.prompt_id}
                    type="button"
                    onClick={() => setManualPromptId(prompt.prompt_id)}
                    className={`border px-3 py-2 text-left text-[12px] transition-colors ${
                      selected
                        ? "border-hud-green bg-hud-green/10 text-foreground"
                        : "border-hud-border bg-black/20 text-hud-dim hover:text-foreground"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-bold">#{prompt.prompt_id}</span>
                      <span className={`border px-1.5 py-0.5 uppercase tracking-[0.14em] ${statusTone(prompt.status === "running" ? "active" : prompt.status)}`}>
                        {prompt.status}
                      </span>
                    </div>
                    <div className="mt-1 truncate text-foreground/80">{prompt.prompt_name}</div>
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
              <div>Fast mode: <span className="text-foreground">{job.fast ? "on" : "off"}</span></div>
              <div>Download LVB: <span className="text-foreground">{job.download_lvb ? "on" : "off"}</span></div>
              <div>Root model: <span className="text-foreground">{shortModel(job.models.root)}</span></div>
              <div>Sub model: <span className="text-foreground">{shortModel(job.models.sub)}</span></div>
              <div>Vision model: <span className="text-foreground">{shortModel(job.models.vision)}</span></div>
              <div>Caption model: <span className="text-foreground">{shortModel(job.models.caption)}</span></div>
              <div>Output dir: <span className="break-all text-foreground">{job.output_dir}</span></div>
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
          {selectedPrompt ? (
            <>
              <div className="min-w-0 border border-hud-border bg-black/20 p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="min-w-0 break-words text-sm font-bold text-foreground">
                        Prompt #{selectedPrompt.prompt_id} · {selectedPrompt.prompt_name}
                      </span>
                      <span className={`border px-2 py-0.5 text-[11px] uppercase tracking-[0.14em] ${statusTone(selectedPrompt.status === "running" ? "active" : selectedPrompt.status)}`}>
                        {selectedPrompt.status}
                      </span>
                    </div>
                    <p className="mt-1 break-words text-[12px] text-hud-dim">{selectedPrompt.question}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-right text-[12px]">
                    <div>
                      <span className="block uppercase tracking-[0.14em] text-hud-dim">Cost</span>
                      <span className="font-bold text-hud-amber">
                        {selectedPrompt.cost_usd != null ? `$${selectedPrompt.cost_usd.toFixed(4)}` : "—"}
                      </span>
                    </div>
                    <div>
                      <span className="block uppercase tracking-[0.14em] text-hud-dim">Time</span>
                      <span className="font-bold text-foreground">{formatTime(selectedPrompt.wall_time_s)}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="min-w-0 overflow-hidden">
                <TraceTimeline
                  events={selectedTrace}
                  startTime={selectedTrace[0]?.timestamp ?? selectedPrompt.started_at ?? null}
                />
              </div>
            </>
          ) : (
            <div className="border border-hud-border bg-black/20 px-4 py-8 text-center text-[12px] uppercase tracking-[0.14em] text-hud-dim">
              No prompt selected
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function BenchmarkLauncher({ onResultsChanged }: { onResultsChanged?: () => void }) {
  const [catalog, setCatalog] = useState<BenchmarkCatalog | null>(null);
  const [jobs, setJobs] = useState<BenchmarkJobSummary[]>([]);
  const [preset, setPreset] = useState<PromptPreset>("all");
  const [selectedPromptIds, setSelectedPromptIds] = useState<number[]>([]);
  const [showPromptPicker, setShowPromptPicker] = useState(false);
  const [form, setForm] = useState<LauncherFormState>({
    workers: 6,
    maxIterations: 20,
    maxDepth: 2,
    maxBudgetUsd: 1.0,
    fast: false,
    runName: "",
    outputDir: "",
    downloadLvb: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmarkCatalog()
      .then((nextCatalog) => {
        setCatalog(nextCatalog);
        setSelectedPromptIds(nextCatalog.defaults.prompt_presets.all);
        setForm({
          workers: nextCatalog.defaults.workers,
          maxIterations: nextCatalog.defaults.max_iterations,
          maxDepth: nextCatalog.defaults.max_depth,
          maxBudgetUsd: nextCatalog.defaults.max_budget_usd,
          fast: nextCatalog.defaults.fast,
          runName: "",
          outputDir: nextCatalog.defaults.output_dir,
          downloadLvb: nextCatalog.defaults.download_lvb,
        });
      })
      .catch((nextError: Error) => setError(nextError.message));

    fetchBenchmarkJobs()
      .then(setJobs)
      .catch((nextError: Error) => setError(nextError.message));
  }, []);

  const promptsByGroup = useMemo(() => {
    if (!catalog) return { demo: [] as BenchmarkPromptCatalogItem[], lvb: [] as BenchmarkPromptCatalogItem[] };
    return {
      demo: catalog.prompts.filter((prompt) => prompt.group === "demo"),
      lvb: catalog.prompts.filter((prompt) => prompt.group === "lvb"),
    };
  }, [catalog]);

  const selectedCount = selectedPromptIds.length;
  const selectedDemoCount = selectedPromptIds.filter((id) => promptsByGroup.demo.some((prompt) => prompt.prompt_id === id)).length;
  const selectedLvbCount = selectedPromptIds.filter((id) => promptsByGroup.lvb.some((prompt) => prompt.prompt_id === id)).length;
  const defaultModels = catalog?.defaults.models ?? null;

  const setPresetSelection = (nextPreset: PromptPreset) => {
    setPreset(nextPreset);
    if (!catalog) return;
    if (nextPreset === "custom") return;
    setSelectedPromptIds(catalog.defaults.prompt_presets[nextPreset]);
  };

  const togglePrompt = (promptId: number) => {
    setPreset("custom");
    setSelectedPromptIds((current) => (
      current.includes(promptId)
        ? current.filter((id) => id !== promptId)
        : [...current, promptId].sort((a, b) => a - b)
    ));
  };

  const updateJob = useCallback((nextJob: BenchmarkJobSummary) => {
    setJobs((current) => {
      const existing = current.some((job) => job.job_id === nextJob.job_id);
      const next = existing
        ? current.map((job) => (job.job_id === nextJob.job_id ? nextJob : job))
        : [nextJob, ...current];
      return [...next].sort((a, b) => b.created_at - a.created_at);
    });
  }, []);

  const handleSettled = useCallback(() => {
    onResultsChanged?.();
  }, [onResultsChanged]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (selectedPromptIds.length === 0) {
      setError("Select at least one benchmark prompt.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const job = await createBenchmarkJob({
        benchmark_type: "video",
        prompt_ids: selectedPromptIds,
        workers: form.workers,
        max_iterations: form.maxIterations,
        max_depth: form.maxDepth,
        max_budget_usd: form.maxBudgetUsd,
        fast: form.fast,
        output_dir: form.outputDir.trim() || null,
        run_name: form.runName.trim() || null,
        download_lvb: form.downloadLvb,
      });
      updateJob(job);
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Failed to create benchmark job";
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
            <h2 className="text-sm font-bold uppercase tracking-[0.18em] text-hud-label">Run Benchmarks</h2>
            <p className="mt-1 text-[12px] text-hud-dim">
              Launch video benchmark batches from the dashboard and inspect traces prompt-by-prompt while they run.
            </p>
          </div>
          <div className="text-right text-[12px] text-hud-dim">
            <div>{jobs.filter((job) => job.status === "running" || job.status === "pending" || job.status === "stopping").length} active jobs</div>
            <div>{jobs.length} total submitted</div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 p-4 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <form onSubmit={handleSubmit} className="min-w-0 space-y-4">
          <div className="border border-hud-border bg-black/20 p-3">
            <FieldLabel>Benchmark Subset</FieldLabel>
            <div className="grid grid-cols-2 gap-2">
              {(["all", "demo", "lvb", "custom"] as PromptPreset[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setPresetSelection(option)}
                  className={`border px-3 py-2 text-[12px] font-bold uppercase tracking-[0.14em] transition-colors ${
                    preset === option
                      ? "border-hud-green bg-hud-green/10 text-foreground"
                      : "border-hud-border text-hud-dim hover:text-foreground"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
            <p className="mt-2 text-[12px] text-hud-dim">
              {selectedCount} prompt{selectedCount === 1 ? "" : "s"} selected
            </p>
          </div>

          <div className="border border-hud-border bg-black/20 p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <FieldLabel>Prompt Selection</FieldLabel>
                <p className="text-[12px] text-hud-dim">
                  {preset === "custom" ? "Custom selection" : `Preset: ${preset}`}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowPromptPicker((current) => !current)}
                className="border border-hud-border px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.14em] text-foreground transition-colors hover:border-hud-green hover:text-hud-green"
              >
                {showPromptPicker ? "Hide Advanced Picker" : "Edit Individual Prompts"}
              </button>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3 text-[12px]">
              <div className="border border-hud-border bg-black/10 px-3 py-2">
                <span className="block uppercase tracking-[0.14em] text-hud-dim">Selected</span>
                <span className="font-bold text-foreground">{selectedCount}</span>
              </div>
              <div className="border border-hud-border bg-black/10 px-3 py-2">
                <span className="block uppercase tracking-[0.14em] text-hud-dim">Demo</span>
                <span className="font-bold text-foreground">{selectedDemoCount}</span>
              </div>
              <div className="border border-hud-border bg-black/10 px-3 py-2">
                <span className="block uppercase tracking-[0.14em] text-hud-dim">LVB</span>
                <span className="font-bold text-foreground">{selectedLvbCount}</span>
              </div>
            </div>
            {showPromptPicker && (
              <div className="mt-4 space-y-3">
                {(["demo", "lvb"] as const).map((group) => (
                  <div key={group}>
                    <span className="mb-2 block text-[11px] font-bold uppercase tracking-[0.14em] text-hud-dim">
                      {group === "demo" ? "Demo Benchmarks" : "LongVideoBench"}
                    </span>
                    <div className="grid gap-2">
                      {promptsByGroup[group].map((prompt) => {
                        const checked = selectedPromptIds.includes(prompt.prompt_id);
                        return (
                          <label
                            key={prompt.prompt_id}
                            className={`flex items-start gap-3 border px-3 py-2 text-sm transition-colors ${
                              checked
                                ? "border-hud-green bg-hud-green/10"
                                : "border-hud-border bg-black/10"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => togglePrompt(prompt.prompt_id)}
                              className="mt-1"
                            />
                            <span>
                              <span className="block text-foreground">
                                #{prompt.prompt_id} · {prompt.prompt_name}
                              </span>
                              <span className="block text-[12px] text-hud-dim">{prompt.video_key}</span>
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
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
              <FieldLabel>Budget / Prompt</FieldLabel>
              <NumberInput value={form.maxBudgetUsd} min={0.01} step={0.01} onChange={(value) => setForm((current) => ({ ...current, maxBudgetUsd: value }))} />
            </div>
            <div className="col-span-2">
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
                <div>Sub model: <span className="text-foreground">{shortModel(defaultModels.sub)}</span></div>
                <div>Vision model: <span className="text-foreground">{shortModel(defaultModels.vision)}</span></div>
                <div>Caption model: <span className="text-foreground">{shortModel(defaultModels.caption)}</span></div>
              </div>
            </div>
          )}

          <div className="grid gap-2">
            <CheckboxRow
              checked={form.fast}
              onChange={(checked) => setForm((current) => ({ ...current, fast: checked }))}
              label="Fast mode"
              hint="Matches the CLI behavior: defaults collapse to 10 iterations and $0.50 when enabled."
            />
            <CheckboxRow
              checked={form.downloadLvb}
              onChange={(checked) => setForm((current) => ({ ...current, downloadLvb: checked }))}
              label="Download missing LVB videos first"
              hint="Useful when the selected subset includes LongVideoBench prompts that are not on disk yet."
            />
          </div>

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
            {submitting ? "Starting…" : "Start Benchmark Run"}
          </button>
        </form>

        <div className="min-w-0 space-y-4">
          {jobs.length === 0 ? (
            <div className="border border-hud-border bg-black/20 px-4 py-10 text-center text-[12px] uppercase tracking-[0.14em] text-hud-dim">
              No benchmark jobs yet
            </div>
          ) : (
            jobs.map((job) => (
              <BenchmarkJobCard
                key={job.job_id}
                job={job}
                onSnapshot={updateJob}
                onSettled={handleSettled}
              />
            ))
          )}
        </div>
      </div>
    </section>
  );
}
