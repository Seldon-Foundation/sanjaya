#!/usr/bin/env python3
"""Run Gemini 3 Flash on the same MMOU questions as a Sanjaya MMOU UI run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARKS_DIR = Path(os.getenv("SANJAYA_BENCHMARKS_DIR", "/Users/lsteno/Developer/GitHub/benchmarks"))
DEFAULT_SOURCE_RUN = "mmou_20260425_193820"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def _load_benchmark_api(benchmarks_dir: Path) -> Any:
    src_dir = benchmarks_dir / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Benchmark repo src directory not found: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from videobench.artifacts import build_run_layout
    from videobench.benchmarks.mmou import (
        MMOU_DATASET_FILE,
        MMOU_EVAL_SPACE,
        download_mmou_metadata,
        evaluate_submission_with_api,
        export_submission_from_predictions,
        predict_mmou,
    )
    from videobench.cache import ResponseCache
    from videobench.config import load_config
    from videobench.models.openrouter import OpenRouterAdapter

    return {
        "build_run_layout": build_run_layout,
        "download_mmou_metadata": download_mmou_metadata,
        "evaluate_submission_with_api": evaluate_submission_with_api,
        "export_submission_from_predictions": export_submission_from_predictions,
        "load_config": load_config,
        "OpenRouterAdapter": OpenRouterAdapter,
        "predict_mmou": predict_mmou,
        "ResponseCache": ResponseCache,
        "MMOU_DATASET_FILE": MMOU_DATASET_FILE,
        "MMOU_EVAL_SPACE": MMOU_EVAL_SPACE,
    }


def _find_source_job(jobs_root: Path, run_name: str) -> Path:
    matches: list[tuple[float, Path]] = []
    for job_file in jobs_root.glob("*/job.json"):
        try:
            payload = json.loads(job_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("run_name") == run_name:
            matches.append((float(payload.get("created_at") or 0.0), job_file.parent))
    if not matches:
        raise FileNotFoundError(f"No Sanjaya MMOU job with run_name={run_name!r} under {jobs_root}")
    return sorted(matches, reverse=True)[0][1]


def _source_question_ids(source_job_dir: Path) -> list[str]:
    job = json.loads((source_job_dir / "job.json").read_text(encoding="utf-8"))
    question_ids = [str(item).strip() for item in job.get("question_ids") or [] if str(item).strip()]
    if not question_ids:
        raise ValueError(f"No question_ids found in {source_job_dir / 'job.json'}")
    return question_ids


def _write_subset_dataset(dataset_path: Path, question_ids: list[str], output_path: Path) -> None:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_id = {str(row.get("question_id")): row for row in rows if isinstance(row, dict)}
    missing = [question_id for question_id in question_ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Question ids not present in MMOU dataset: {missing[:20]}")
    selected = [by_id[question_id] for question_id in question_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")


def _default_config(benchmarks_dir: Path) -> Path | None:
    candidate = benchmarks_dir / "configs" / "vertex-benchmark.toml"
    return candidate if candidate.exists() else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gemini 3 Flash on the same 10 MMOU questions as a Sanjaya run.")
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--source-jobs-root", type=Path, default=PROJECT_ROOT / "sanjaya_artifacts" / "mmou_jobs")
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run", default=None)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "sanjaya_artifacts" / "mmou_single_model" / "runs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    api = _load_benchmark_api(args.benchmarks_dir)
    cfg = api["load_config"](args.config or _default_config(args.benchmarks_dir), cwd=args.benchmarks_dir)

    source_job_dir = _find_source_job(args.source_jobs_root, args.source_run)
    question_ids = _source_question_ids(source_job_dir)
    dataset_path = cfg.storage.data_dir / "mmou" / api["MMOU_DATASET_FILE"]
    if not dataset_path.exists():
        api["download_mmou_metadata"](dataset_path.parent, include_captions=False)
    if not dataset_path.exists():
        raise FileNotFoundError(f"MMOU dataset file not found: {dataset_path}")

    run_name = args.run or f"{args.source_run}-gemini-3-flash"
    layout = api["build_run_layout"](args.runs_root, "mmou", run_name)
    subset_path = layout.root / "input_questions.json"
    _write_subset_dataset(dataset_path, question_ids, subset_path)

    adapter = api["OpenRouterAdapter"](
        settings=cfg.openrouter,
        cache=api["ResponseCache"](cfg.storage.cache_db),
    )
    layout.write_manifest(
        {
            "benchmark": "mmou",
            "run": run_name,
            "source_run": args.source_run,
            "source_job_dir": str(source_job_dir),
            "question_ids": question_ids,
            "model": args.model,
            "dataset_path": str(subset_path),
            "video_source": "url",
            "clip_mode": "full",
        }
    )

    summary = api["predict_mmou"](
        adapter=adapter,
        model_name=args.model,
        dataset_path=subset_path,
        layout=layout,
        video_source="url",
        video_dir=None,
        selection_mode="all",
        limit=None,
        clip_mode="full",
        use_video_summary_context=False,
        max_attempts=args.max_attempts,
        continue_on_error=True,
        workers=args.workers,
        force_thread_workers=True,
    )
    exported = api["export_submission_from_predictions"](
        layout.records_dir / "predictions.jsonl",
        layout.submissions_dir,
        stem=run_name,
    )
    evaluation = None
    if not args.skip_evaluate:
        evaluation = api["evaluate_submission_with_api"](
            submission_file=Path(exported["json_path"]),
            output_dir=layout.judge_dir,
            evaluator_space=api["MMOU_EVAL_SPACE"],
            hf_token=os.getenv(args.hf_token_env) or None,
        )

    result = {
        "run": run_name,
        "layout": str(layout.root),
        "source_job_dir": str(source_job_dir),
        "model": args.model,
        "summary": summary,
        "exported": exported,
        "evaluation": evaluation,
    }
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
