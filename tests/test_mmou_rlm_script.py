from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module() -> Any:
    script_path = ROOT / "scripts" / "run_mmou_rlm.py"
    spec = importlib.util.spec_from_file_location("run_mmou_rlm", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_mmou_rlm_wires_videobench_url_mode_export_and_evaluation(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    calls: dict[str, Any] = {}

    class FakeLayout:
        def __init__(self, runs_dir: Path, benchmark: str, run: str) -> None:
            self.root = runs_dir / benchmark / run
            self.records_dir = self.root / "records"
            self.submissions_dir = self.root / "submissions"
            self.judge_dir = self.root / "judge"
            self.records_dir.mkdir(parents=True)
            self.submissions_dir.mkdir(parents=True)
            self.judge_dir.mkdir(parents=True)

        def write_manifest(self, payload: dict[str, Any]) -> None:
            calls["manifest"] = payload

    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"

    def fake_load_config(config: Path | None, cwd: Path) -> Any:
        calls["load_config"] = {"config": config, "cwd": cwd}
        return SimpleNamespace(storage=SimpleNamespace(data_dir=data_dir, runs_dir=runs_dir))

    def fake_download_mmou_metadata(dataset_root: Path, include_captions: bool = False) -> dict[str, Any]:
        calls["download"] = {"dataset_root": dataset_root, "include_captions": include_captions}
        dataset_root.mkdir(parents=True)
        (dataset_root / "MMOU.json").write_text("[]", encoding="utf-8")
        return {"dataset_root": str(dataset_root)}

    def fake_build_run_layout(runs_root: Path, benchmark: str, run: str) -> FakeLayout:
        calls["layout"] = {"runs_root": runs_root, "benchmark": benchmark, "run": run}
        return FakeLayout(runs_root, benchmark, run)

    def fake_predict_mmou(**kwargs: Any) -> dict[str, Any]:
        calls["predict"] = kwargs
        return {"valid_ready_predictions": 1}

    def fake_export(records_path: Path, output_dir: Path, stem: str) -> dict[str, Any]:
        calls["export"] = {"records_path": records_path, "output_dir": output_dir, "stem": stem}
        return {"json_path": str(output_dir / f"{stem}.json"), "rows": 1}

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls["evaluate"] = kwargs
        return {"markdown_outputs": ["Accuracy: 100%"]}

    fake_api = SimpleNamespace(
        build_run_layout=fake_build_run_layout,
        download_mmou_metadata=fake_download_mmou_metadata,
        evaluate_submission_with_api=fake_evaluate,
        export_submission_from_predictions=fake_export,
        load_config=fake_load_config,
        predict_mmou=fake_predict_mmou,
        MMOU_DATASET_FILE="MMOU.json",
        MMOU_EVAL_SPACE="nvidia/MMOU-Eval",
    )
    monkeypatch.setattr(module, "_load_benchmark_api", lambda benchmarks_dir: fake_api)

    result = module.main(
        [
            "--benchmarks-dir",
            str(tmp_path / "benchmarks"),
            "--run",
            "demo",
            "--limit",
            "5",
            "--domains",
            "Sports,News",
            "--per-domain-limit",
            "2",
            "--workers",
            "1",
            "--max-attempts",
            "2",
            "--root-model",
            "root",
            "--sub-model",
            "flash",
            "--recursive-model",
            "pro-child",
            "--vision-model",
            "vision",
            "--audio-model",
            "audio",
            "--keep-sanjaya-artifacts",
            "--sanjaya-artifacts-dir",
            str(tmp_path / "sanjaya-artifacts"),
            "--evaluate",
        ]
    )

    predict = calls["predict"]
    assert result["run"] == "demo"
    assert calls["load_config"]["cwd"] == tmp_path / "benchmarks"
    assert calls["download"]["dataset_root"] == data_dir / "mmou"
    assert calls["layout"] == {"runs_root": runs_dir, "benchmark": "mmou", "run": "demo"}
    assert calls["manifest"]["video_cache"] == "videobench_s3_youtube_cache"
    assert calls["manifest"]["clip_mode"] == "full"
    assert calls["manifest"]["sanjaya"]["recursive_model"] == "pro-child"

    assert predict["video_source"] == "url"
    assert predict["video_dir"] is None
    assert predict["clip_mode"] == "full"
    assert predict["use_video_summary_context"] is False
    assert predict["domains"] == ["Sports", "News"]
    assert predict["limit"] == 5
    assert predict["per_domain_limit"] == 2
    assert predict["max_attempts"] == 2
    assert predict["continue_on_error"] is True
    assert predict["workers"] == 1
    assert predict["force_thread_workers"] is True
    assert predict["adapter"].root_model == "root"
    assert predict["adapter"].sub_model == "flash"
    assert predict["adapter"].recursive_model == "pro-child"
    assert predict["adapter"].keep_artifacts is True

    assert calls["export"]["records_path"] == runs_dir / "mmou" / "demo" / "records" / "predictions.jsonl"
    assert calls["export"]["stem"] == "submission"
    assert calls["evaluate"]["submission_file"] == runs_dir / "mmou" / "demo" / "submissions" / "submission.json"
    assert calls["evaluate"]["evaluator_space"] == "nvidia/MMOU-Eval"
