from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "api"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sanjaya_api.models import MMOUJobCreateRequest
from sanjaya_api.services import mmou_jobs
from sanjaya_api.services.mmou_jobs import MMOUBenchmarkJobService, select_mmou_samples_for_ui


class FakeThread:
    def __init__(self, target=None, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        return

    def is_alive(self) -> bool:
        return False


class FakeSample:
    def __init__(
        self,
        question_id: str,
        domain: str,
        *,
        question: str | None = None,
        options: dict[str, str] | None = None,
        video_url: str | None = None,
        subdomain: str = "sub",
        question_type: list[str] | None = None,
        start_time: str = "00:00",
        end_time: str = "00:05",
        video_duration: float | None = 10.0,
    ):
        self.question_id = question_id
        self.question = question or f"Question {question_id}?"
        self.options = options or {"A": "Alpha", "B": "Beta"}
        self.video_url = video_url or f"https://youtube.com/watch?v={question_id}"
        self.domain = domain
        self.subdomain = subdomain
        self.question_type = question_type or ["qa"]
        self.start_time = start_time
        self.end_time = end_time
        self.video_duration = video_duration


class FakeGenerationRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeMediaInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeMediaType:
    video = "video"


class FakeTracer:
    def __init__(self):
        self.events = [
            {"kind": "sanjaya.completion_start", "timestamp": 1.0, "model": "root"},
            {"kind": "sanjaya.completion_end", "timestamp": 2.0, "status": "complete"},
        ]

    def dump_events(self):
        return list(self.events)


class FakeAgent:
    def __init__(self):
        self._tracer = FakeTracer()


class FakeAdapter:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeAdapter.calls.append(kwargs)

    def generate(self, request):
        callback = self.kwargs.get("agent_callback")
        if callback:
            callback(FakeAgent(), {"run_id": "fake-run"})
        return SimpleNamespace(
            text='{"answer": "B"}',
            usage={"iterations": 3, "cost_usd": 0.25, "wall_time_s": 4.0},
            cache_key="cache-key",
            cached=False,
        )


@pytest.fixture
def samples() -> list[FakeSample]:
    return [
        FakeSample("a1", "A"),
        FakeSample("a2", "A"),
        FakeSample("b1", "B"),
        FakeSample("b2", "B"),
        FakeSample("c1", "C"),
    ]


@pytest.fixture
def fake_api(samples: list[FakeSample], tmp_path: Path):
    def download_remote_video(sample, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        path = destination_dir / f"{sample.question_id}.mp4"
        path.write_bytes(b"video")
        downloaded_dirs.append(destination_dir)
        return path

    def export_submission_from_predictions(predictions_path: Path, output_dir: Path, stem: str):
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ready = [
            {"question_id": row["question_id"], "answer": row["answer"]}
            for row in rows
            if row.get("question_id") and row.get("answer")
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{stem}.json"
        jsonl_path = output_dir / f"{stem}.jsonl"
        json_path.write_text(json.dumps(ready), encoding="utf-8")
        jsonl_path.write_text("\n".join(json.dumps(row) for row in ready), encoding="utf-8")
        exported_submissions.append(ready)
        return {"json_path": str(json_path), "jsonl_path": str(jsonl_path), "rows": len(ready)}

    def evaluate_submission_with_api(submission_file: Path, output_dir: Path, evaluator_space: str, hf_token=None):
        submitted = json.loads(submission_file.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "evaluator_space": evaluator_space,
            "evaluated_at": "2026-04-26T00:00:00+00:00",
            "submission_file": str(submission_file),
            "markdown_outputs": [
                "### Metrics\n- Official accuracy: `0.01%` (`1 / 15000`)\n- Answered accuracy: `50.00%` (`1 / 2`)"
            ],
            "domain_breakdown": {"headers": [], "data": [], "metadata": None},
            "duration_breakdown": {
                "headers": [
                    "Duration Bucket",
                    "Official Accuracy (%)",
                    "Answered Accuracy (%)",
                    "Coverage (%)",
                    "Correct",
                    "Answered",
                    "Total",
                ],
                "data": [["Overall", 0.01, 50.0, 0.02, 1, len(submitted), 15000]],
                "metadata": None,
            },
            "skill_breakdown": {"headers": [], "data": [], "metadata": None},
        }
        (output_dir / "mmou_eval_result.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    downloaded_dirs: list[Path] = []
    exported_submissions: list[list[dict[str, str]]] = []
    api = SimpleNamespace(
        MMOU_DATASET_FILE="MMOU.json",
        MMOU_EVAL_SPACE="nvidia/MMOU-Eval",
        MMOUSample=FakeSample,
        build_mmou_prompt=lambda sample: f"Prompt {sample.question_id}",
        download_mmou_metadata=lambda dataset_root, include_captions=False: None,
        download_remote_video=download_remote_video,
        downloaded_dirs=downloaded_dirs,
        evaluate_submission_with_api=evaluate_submission_with_api,
        exported_submissions=exported_submissions,
        export_submission_from_predictions=export_submission_from_predictions,
        GenerationRequest=FakeGenerationRequest,
        load_config=lambda _config, cwd: SimpleNamespace(storage=SimpleNamespace(data_dir=tmp_path / "data")),
        load_mmou_dataset=lambda _path: samples,
        MediaInput=FakeMediaInput,
        MediaType=FakeMediaType,
        mmou_domain_counts=lambda rows: {"A": 2, "B": 2, "C": 1},
        parse_answer_letter=lambda text: json.loads(text)["answer"],
    )
    dataset = tmp_path / "data" / "mmou" / "MMOU.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("[]", encoding="utf-8")
    return api


def test_select_mmou_samples_stratifies_exact_limit(samples: list[FakeSample]) -> None:
    selected, source = select_mmou_samples_for_ui(samples, limit=5, stratified=True)

    assert source == "dataset"
    assert [sample.question_id for sample in selected] == ["a1", "b1", "c1", "a2", "b2"]


def test_question_ids_dedupe_and_preserve_file_order_with_limit(samples: list[FakeSample]) -> None:
    selected, source = select_mmou_samples_for_ui(
        samples,
        limit=2,
        stratified=True,
        question_ids=["b2", "a1", "b2", "c1"],
    )

    assert source == "question_ids"
    assert [sample.question_id for sample in selected] == ["b2", "a1"]


def test_unknown_question_ids_are_rejected(samples: list[FakeSample]) -> None:
    with pytest.raises(ValueError, match="Unknown MMOU question ids"):
        select_mmou_samples_for_ui(samples, limit=1, stratified=False, question_ids=["missing"])


def test_start_job_persists_selection_and_uses_question_ids(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    summary = service.start_job(MMOUJobCreateRequest(
        limit=1,
        question_ids=["b2", "a1"],
    ))

    assert summary.selection_source == "question_ids"
    assert summary.question_ids == ["b2"]
    assert Path(summary.job_dir, "job.json").exists()
    assert Path(summary.job_dir, "questions.json").exists()


def test_run_job_writes_predictions_trace_and_cleans_temp_video_dir(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    real_thread = mmou_jobs.threading.Thread
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)
    monkeypatch.setattr(mmou_jobs, "SanjayaMMOUAdapter", FakeAdapter)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    created = service.start_job(MMOUJobCreateRequest(limit=1, stratified=False))
    record = service.get_job_record(created.job_id)
    assert record is not None

    monkeypatch.setattr(mmou_jobs.threading, "Thread", real_thread)
    service._run_job(record)
    summary = service.get_job(created.job_id)
    assert summary is not None
    assert summary.completed_questions == 1
    assert summary.questions[0].answer == "B"

    predictions = Path(summary.job_dir, "records", "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(predictions) == 1
    assert json.loads(predictions[0])["answer"] == "B"

    trace_payload = json.loads(Path(summary.questions[0].trace_path).read_text(encoding="utf-8"))
    assert trace_payload["events"][0]["kind"] == "run_start"
    assert fake_api.downloaded_dirs
    assert all(not path.exists() for path in fake_api.downloaded_dirs)
    assert FakeAdapter.calls[-1]["recursive_model"] == "google-vertex:gemini-3.1-pro-preview"


def test_resume_skips_completed_predictions(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    created = service.start_job(MMOUJobCreateRequest(limit=2, stratified=False))
    record = service.get_job_record(created.job_id)
    assert record is not None
    first = record.questions[record.question_ids[0]]
    first.prediction_row = {
        "question_id": first.question_id,
        "answer": "A",
    }
    service._persist_predictions(record)
    record.status = "error"
    second = record.questions[record.question_ids[1]]
    second.status = "error"

    resumed = service.resume_job(created.job_id)

    assert resumed is not None
    assert resumed.status == "pending"
    assert resumed.questions[0].status == "complete"
    assert resumed.questions[1].status == "pending"


def test_evaluate_job_scores_only_answered_predictions_and_persists_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    created = service.start_job(MMOUJobCreateRequest(limit=3, stratified=False))
    record = service.get_job_record(created.job_id)
    assert record is not None
    first = record.questions[record.question_ids[0]]
    second = record.questions[record.question_ids[1]]
    first.prediction_row = {"question_id": first.question_id, "answer": "A"}
    second.prediction_row = {"question_id": second.question_id, "answer": "B"}
    record.questions[record.question_ids[2]].prediction_row = {
        "question_id": record.question_ids[2],
        "answer": None,
        "error": "not answered",
    }
    record.status = "running"
    service._persist_predictions(record)

    summary = service.evaluate_job(created.job_id)

    assert summary is not None
    assert summary.answered_accuracy_pct == 50.0
    assert summary.correct == 1
    assert summary.answered == 2
    assert summary.submission_rows == 2
    assert fake_api.exported_submissions == [[
        {"question_id": first.question_id, "answer": "A"},
        {"question_id": second.question_id, "answer": "B"},
    ]]
    persisted = json.loads(Path(record.job_dir, "judge", "mmou_eval_summary.json").read_text(encoding="utf-8"))
    assert persisted["answered_accuracy_pct"] == 50.0

    rehydrated = MMOUBenchmarkJobService(output_root=tmp_path)
    jobs = rehydrated.list_jobs()
    assert jobs[0].latest_evaluation is not None
    assert jobs[0].latest_evaluation.answered == 2


def test_evaluate_job_rejects_empty_answered_predictions(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    created = service.start_job(MMOUJobCreateRequest(limit=1, stratified=False))

    with pytest.raises(ValueError, match="No answered MMOU predictions"):
        service.evaluate_job(created.job_id)


def test_evaluate_question_scores_one_answer_and_persists_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    created = service.start_job(MMOUJobCreateRequest(limit=2, stratified=False))
    record = service.get_job_record(created.job_id)
    assert record is not None
    first = record.questions[record.question_ids[0]]
    second = record.questions[record.question_ids[1]]
    first.prediction_row = {"question_id": first.question_id, "answer": "A"}
    second.prediction_row = {"question_id": second.question_id, "answer": "B"}
    first.status = "complete"
    second.status = "complete"
    service._persist_predictions(record)

    summary = service.evaluate_question(created.job_id, second.question_id)

    assert summary is not None
    assert summary.question_id == second.question_id
    assert summary.answer == "B"
    assert summary.correct is True
    assert summary.submission_rows == 1
    assert fake_api.exported_submissions[-1] == [{"question_id": second.question_id, "answer": "B"}]

    persisted = json.loads(
        Path(record.job_dir, "judge", "questions", f"{second.question_id}.json").read_text(encoding="utf-8")
    )
    assert persisted["correct"] is True

    rehydrated = MMOUBenchmarkJobService(output_root=tmp_path)
    jobs = rehydrated.list_jobs()
    assert jobs[0].questions[1].latest_evaluation is not None
    assert jobs[0].questions[1].latest_evaluation.correct is True


def test_evaluate_question_rejects_unanswered_prediction(
    monkeypatch: pytest.MonkeyPatch,
    fake_api,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mmou_jobs, "load_mmou_benchmark_api", lambda _benchmarks_dir=None: fake_api)
    monkeypatch.setattr(mmou_jobs.threading, "Thread", FakeThread)

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    created = service.start_job(MMOUJobCreateRequest(limit=1, stratified=False))

    with pytest.raises(ValueError, match="No answered MMOU prediction"):
        service.evaluate_question(created.job_id, "a1")


def test_hydration_marks_abandoned_running_job_interrupted(tmp_path: Path) -> None:
    job_dir = tmp_path / "mmou_job_saved"
    job_dir.mkdir(parents=True)
    request = MMOUJobCreateRequest(limit=1).model_dump()
    (job_dir / "job.json").write_text(
        json.dumps({
            "job_id": "mmou_job_saved",
            "request": request,
            "run_name": "saved",
            "output_dir": str(tmp_path),
            "job_dir": str(job_dir),
            "models": {},
            "question_ids": ["q1"],
            "selection_source": "dataset",
            "status": "running",
            "created_at": 1.0,
            "revision": 0,
        }),
        encoding="utf-8",
    )
    (job_dir / "questions.json").write_text(
        json.dumps([
            {
                "question_id": "q1",
                "question": "Question?",
                "options": {"A": "Alpha"},
                "video_url": "https://youtube.com/watch?v=q1",
                "domain": "A",
                "subdomain": "sub",
                "question_type": ["qa"],
                "start_time": "00:00",
                "end_time": "00:01",
                "video_duration": 1.0,
                "status": "running",
            }
        ]),
        encoding="utf-8",
    )

    service = MMOUBenchmarkJobService(output_root=tmp_path)
    jobs = service.list_jobs()

    assert len(jobs) == 1
    assert jobs[0].status == "interrupted"
    assert jobs[0].questions[0].status == "error"
