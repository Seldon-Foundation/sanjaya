from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "api"))

from sanjaya_api.models import BenchmarkJobCreateRequest
from sanjaya_api.services import benchmark_jobs
from sanjaya_api.services.benchmark_jobs import BenchmarkJobService


class FakeThread:
    def __init__(self, target=None, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        return


@pytest.fixture
def fake_video_module(tmp_path: Path):
    demo_video = tmp_path / "demo.mp4"
    lvb_video = tmp_path / "lvb.mp4"
    demo_video.write_bytes(b"demo")
    lvb_video.write_bytes(b"lvb")

    return SimpleNamespace(
        ROOT_MODEL="root-model",
        SUB_MODEL="sub-model",
        VISION_MODEL="vision-model",
        CAPTION_MODEL="caption-model",
        CRITIC_MODEL=None,
        RESULTS_DIR=tmp_path / "results",
        PROMPTS=[
            {"id": 1, "name": "demo_prompt", "video_key": "demo_video", "question": "Summarize"},
            {"id": 13, "name": "lvb_prompt", "video_key": "lvb_video", "question": "MCQ", "is_mcq": True},
        ],
        VIDEOS={
            "demo_video": {"video": str(demo_video)},
            "lvb_video": {"video": str(lvb_video)},
        },
        LVB_QUESTIONS={
            13: {
                "question": "What color is the bus?",
                "candidates": ["Red", "Blue"],
                "gt_answer": "Red",
            },
        },
        download_lvb_videos=lambda: None,
        _check_subtitle_exists=lambda _path: False,
        _format_mcq_question=lambda question, candidates: f"{question} :: {' / '.join(candidates)}",
    )


def test_catalog_exposes_groups_and_mcq_question(monkeypatch: pytest.MonkeyPatch, fake_video_module) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)

    service = BenchmarkJobService()
    catalog = service.get_catalog()

    assert catalog.benchmark_type == "video"
    assert catalog.defaults["prompt_presets"]["all"] == [1, 13]
    assert catalog.prompts[0].group == "demo"
    assert catalog.prompts[1].group == "lvb"
    assert catalog.prompts[1].question == "What color is the bus? :: Red / Blue"


def test_start_job_uses_fast_defaults_and_selected_prompt_ids(
    monkeypatch: pytest.MonkeyPatch,
    fake_video_module,
) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)
    monkeypatch.setattr(benchmark_jobs.threading, "Thread", FakeThread)
    monkeypatch.setattr(benchmark_jobs, "_is_valid_video_path", lambda _path: (True, None))

    service = BenchmarkJobService()
    summary = service.start_job(BenchmarkJobCreateRequest(
        prompt_ids=[13],
        fast=True,
    ))

    assert summary.prompt_ids == [13]
    assert summary.max_iterations == 10
    assert summary.max_budget_usd == 0.5
    assert summary.prompts[0].prompt_id == 13
    assert summary.prompts[0].status == "pending"


def test_start_job_rejects_unknown_prompt_ids(monkeypatch: pytest.MonkeyPatch, fake_video_module) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)

    service = BenchmarkJobService()

    with pytest.raises(ValueError, match="Unknown prompt ids"):
        service.start_job(BenchmarkJobCreateRequest(prompt_ids=[999]))


def test_request_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch, fake_video_module) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)
    monkeypatch.setattr(benchmark_jobs.threading, "Thread", FakeThread)
    monkeypatch.setattr(benchmark_jobs, "_is_valid_video_path", lambda _path: (True, None))

    service = BenchmarkJobService()
    created = service.start_job(BenchmarkJobCreateRequest(prompt_ids=[1, 13]))

    stopped_once = service.request_stop(created.job_id)
    stopped_twice = service.request_stop(created.job_id)

    assert stopped_once is not None
    assert stopped_once.status == "stopping"
    assert stopped_once.stop_requested_at is not None
    assert stopped_twice is not None
    assert stopped_twice.status == "stopping"
    assert stopped_twice.stop_requested_at == stopped_once.stop_requested_at


def test_run_job_marks_unstarted_prompts_stopped_after_stop_request(
    monkeypatch: pytest.MonkeyPatch,
    fake_video_module,
) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)
    monkeypatch.setattr(benchmark_jobs.threading, "Thread", FakeThread)
    monkeypatch.setattr(benchmark_jobs, "_is_valid_video_path", lambda _path: (True, None))

    service = BenchmarkJobService()
    created = service.start_job(BenchmarkJobCreateRequest(prompt_ids=[1, 13]))
    summary = service.request_stop(created.job_id)
    assert summary is not None

    record = service.get_job_record(created.job_id)
    assert record is not None

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_run_single_prompt should not be called after stop request before scheduling")

    monkeypatch.setattr(service, "_run_single_prompt", fail_if_called)

    service._run_job(record)
    final = service.get_job(created.job_id)

    assert final is not None
    assert final.status == "stopped"
    assert final.stop_requested_at is not None
    assert all(prompt.status == "stopped" for prompt in final.prompts)


def test_request_stop_returns_terminal_job_unchanged(monkeypatch: pytest.MonkeyPatch, fake_video_module) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)
    monkeypatch.setattr(benchmark_jobs.threading, "Thread", FakeThread)

    service = BenchmarkJobService()
    created = service.start_job(BenchmarkJobCreateRequest(prompt_ids=[1]))
    record = service.get_job_record(created.job_id)
    assert record is not None
    record.status = "complete"

    summary = service.request_stop(created.job_id)

    assert summary is not None
    assert summary.status == "complete"
    assert summary.stop_requested_at is None


def test_run_job_marks_invalid_existing_video_as_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_video_module,
) -> None:
    monkeypatch.setattr(benchmark_jobs, "load_video_benchmark_module", lambda: fake_video_module)
    monkeypatch.setattr(benchmark_jobs.threading, "Thread", FakeThread)

    def fake_validate(path: Path) -> tuple[bool, str | None]:
        if path.name == "demo.mp4":
            return False, "Could not parse duration: None"
        return True, None

    monkeypatch.setattr(benchmark_jobs, "_is_valid_video_path", fake_validate)

    service = BenchmarkJobService()
    created = service.start_job(BenchmarkJobCreateRequest(prompt_ids=[1, 13]))
    record = service.get_job_record(created.job_id)
    assert record is not None

    def fake_run_single_prompt(_record, _prompt_record, prompt):
        return {
            "prompt_id": prompt["id"],
            "prompt_name": prompt["name"],
            "video_key": prompt["video_key"],
            "question": prompt["question"],
            "is_mcq": bool(prompt.get("is_mcq", False)),
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_time_s": 0,
            "iterations": 0,
            "evidence_count": 0,
            "evidence_sources": [],
            "subtitle": {"had_existing_subtitle": False, "subtitle_generated": False, "subtitle_source": "none"},
            "answer_text": "",
            "answer_data": None,
            "trace_events": [],
        }

    monkeypatch.setattr(service, "_run_single_prompt", fake_run_single_prompt)

    service._run_job(record)
    final = service.get_job(created.job_id)

    assert final is not None
    demo_prompt = next(prompt for prompt in final.prompts if prompt.prompt_id == 1)
    assert demo_prompt.status == "error"
    assert demo_prompt.error == "Could not parse duration: None"
