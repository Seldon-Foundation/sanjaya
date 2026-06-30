"""CLI entrypoint for running MMOU with Sanjaya."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sanjaya.model_defaults import DEFAULT_AUDIO_MODEL, DEFAULT_ROOT_MODEL, DEFAULT_SUB_MODEL, DEFAULT_VISION_MODEL

from .mmou_adapter import SanjayaMMOUAdapter

DEFAULT_BENCHMARKS_DIR = Path(os.getenv("SANJAYA_BENCHMARKS_DIR", "/Users/lsteno/Developer/GitHub/benchmarks"))


def _load_benchmark_api(benchmarks_dir: Path) -> SimpleNamespace:
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
    from videobench.config import load_config

    return SimpleNamespace(
        MMOU_DATASET_FILE=MMOU_DATASET_FILE,
        MMOU_EVAL_SPACE=MMOU_EVAL_SPACE,
        build_run_layout=build_run_layout,
        download_mmou_metadata=download_mmou_metadata,
        evaluate_submission_with_api=evaluate_submission_with_api,
        export_submission_from_predictions=export_submission_from_predictions,
        load_config=load_config,
        predict_mmou=predict_mmou,
    )


def _parse_domains(value: str) -> list[str] | None:
    if value.strip().lower() == "all":
        return None
    domains = [part.strip() for part in value.split(",") if part.strip()]
    return domains or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMOU with the Sanjaya video agent.")
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--config", type=Path, default=None, help="Optional videobench TOML config.")
    parser.add_argument("--run", default=None, help="Run name under the videobench MMOU artifacts directory.")
    parser.add_argument("--dataset-file", type=Path, default=None, help="Override MMOU metadata JSON file.")
    parser.add_argument("--domains", default="all", help="Comma-separated domain filter or 'all'.")
    parser.add_argument("--selection-mode", default="balanced_domains")
    parser.add_argument("--per-domain-limit", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="Optional total question limit.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--model-name", default="sanjaya-rlm")
    parser.add_argument("--root-model", default=DEFAULT_ROOT_MODEL)
    parser.add_argument("--sub-model", default=DEFAULT_SUB_MODEL)
    parser.add_argument("--recursive-model", default=DEFAULT_ROOT_MODEL)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--audio-model", default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--max-timeout-s", type=float, default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--submission-stem", default="submission")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluator-space", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    benchmarks_dir = args.benchmarks_dir.expanduser().resolve()
    api = _load_benchmark_api(benchmarks_dir)
    cfg = api.load_config(args.config, cwd=benchmarks_dir)

    run_name = args.run or f"sanjaya-mmou-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    dataset_path = args.dataset_file or (cfg.storage.data_dir / "mmou" / api.MMOU_DATASET_FILE)
    if not dataset_path.exists():
        api.download_mmou_metadata(dataset_path.parent, include_captions=False)
    if not dataset_path.exists():
        raise FileNotFoundError(f"MMOU dataset file not found: {dataset_path}")

    layout = api.build_run_layout(cfg.storage.runs_dir, "mmou", run_name)
    adapter = SanjayaMMOUAdapter(
        name=args.model_name,
        root_model=args.root_model,
        sub_model=args.sub_model,
        recursive_model=args.recursive_model,
        vision_model=args.vision_model,
        audio_model=args.audio_model,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        max_budget_usd=args.max_budget_usd,
        max_timeout_s=args.max_timeout_s,
        keep_artifacts=args.keep_artifacts,
        artifacts_root=args.artifacts_dir,
    )

    layout.write_manifest(
        {
            "benchmark": "mmou",
            "run": run_name,
            "model": args.model_name,
            "adapter": "sanjaya-rlm",
            "dataset_path": str(dataset_path),
            "selection_mode": args.selection_mode,
            "domains": args.domains,
            "limit": args.limit,
            "workers": args.workers,
            "sanjaya": {
                "root_model": args.root_model,
                "sub_model": args.sub_model,
                "recursive_model": args.recursive_model,
                "vision_model": args.vision_model,
                "audio_model": args.audio_model,
                "max_iterations": args.max_iterations,
                "max_depth": args.max_depth,
                "max_budget_usd": args.max_budget_usd,
                "max_timeout_s": args.max_timeout_s,
            },
        }
    )

    summary = api.predict_mmou(
        adapter=adapter,
        model_name=args.model_name,
        dataset_path=dataset_path,
        layout=layout,
        video_source="url",
        video_dir=None,
        selection_mode=args.selection_mode,
        per_domain_limit=args.per_domain_limit,
        domains=_parse_domains(args.domains),
        limit=args.limit,
        clip_mode="full",
        use_video_summary_context=False,
        max_attempts=args.max_attempts,
        continue_on_error=not args.fail_fast,
        workers=args.workers,
        force_thread_workers=True,
    )

    exported: dict[str, Any] | None = None
    if not args.skip_export or args.evaluate:
        exported = api.export_submission_from_predictions(
            layout.records_dir / "predictions.jsonl",
            layout.submissions_dir,
            stem=args.submission_stem,
        )

    evaluation: dict[str, Any] | None = None
    if args.evaluate:
        if exported is None:
            raise ValueError("--evaluate requires exported predictions.")
        evaluation = api.evaluate_submission_with_api(
            submission_file=Path(exported["json_path"]),
            output_dir=layout.judge_dir,
            evaluator_space=args.evaluator_space or api.MMOU_EVAL_SPACE,
            hf_token=os.getenv(args.hf_token_env) or None,
        )

    result = {"run": run_name, "summary": summary, "exported": exported, "evaluation": evaluation}
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
