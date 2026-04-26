from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "api"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sanjaya_api.models import MMOUCatalogResponse, MMOUJobSummary, MMOUQuestionTraceResponse
from sanjaya_api.routes import mmou_jobs as mmou_routes


def _summary(status: str = "pending") -> MMOUJobSummary:
    return MMOUJobSummary(
        job_id="mmou_job_test",
        status=status,  # type: ignore[arg-type]
        created_at=1.0,
        started_at=None,
        finished_at=None,
        stop_requested_at=None,
        stop_reason=None,
        run_name="test",
        output_dir="/tmp/mmou",
        job_dir="/tmp/mmou/mmou_job_test",
        models={"root": "root", "sub": "sub", "recursive": "recursive", "vision": "vision", "audio": "audio"},
        workers=1,
        max_iterations=8,
        max_depth=2,
        max_budget_usd=None,
        max_timeout_s=None,
        limit=1,
        stratified=True,
        domains=None,
        selection_source="dataset",
        keep_artifacts=False,
        total_questions=1,
        completed_questions=0,
        error_questions=0,
        active_question_ids=[],
        question_ids=["q1"],
        questions=[
            {
                "question_id": "q1",
                "question": "Question?",
                "options": {"A": "Alpha"},
                "domain": "A",
                "subdomain": "sub",
                "question_type": ["qa"],
                "start_time": "00:00",
                "end_time": "00:01",
                "status": "pending",
            }
        ],
        stdout_tail=[],
        stderr_tail=[],
        revision=1,
    )


class FakeMMOUService:
    def __init__(self):
        self.created_payloads = []
        self.resumed = []

    def get_catalog(self):
        return MMOUCatalogResponse(
            total_questions=1,
            domain_counts={"A": 1},
            defaults={"limit": 1},
        )

    def start_job(self, request):
        self.created_payloads.append(request)
        return _summary()

    def list_jobs(self):
        return [_summary()]

    def get_job(self, job_id):
        return _summary() if job_id == "mmou_job_test" else None

    def request_stop(self, job_id):
        return _summary("stopping") if job_id == "mmou_job_test" else None

    def resume_job(self, job_id):
        self.resumed.append(job_id)
        return _summary("pending") if job_id == "mmou_job_test" else None

    def get_question_trace(self, job_id, question_id):
        if job_id != "mmou_job_test" or question_id != "q1":
            return None
        response = MMOUQuestionTraceResponse(
            question_id="q1",
            run_id="run-q1",
            events=[{"kind": "run_start", "timestamp": 1.0, "payload": {}}],
        )
        return response.run_id, [event.model_dump() for event in response.events]


def test_mmou_routes_create_list_resume_and_trace(monkeypatch) -> None:
    service = FakeMMOUService()
    monkeypatch.setattr(mmou_routes, "_mmou_jobs", service)
    app = FastAPI()
    app.include_router(mmou_routes.router)
    client = TestClient(app)

    created = client.post("/mmou-jobs", json={"limit": 1, "question_ids": ["q1"]})
    listed = client.get("/mmou-jobs")
    resumed = client.post("/mmou-jobs/mmou_job_test/resume")
    trace = client.get("/mmou-jobs/mmou_job_test/questions/q1/trace")

    assert created.status_code == 200
    assert created.json()["job_id"] == "mmou_job_test"
    assert service.created_payloads[0].question_ids == ["q1"]
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert resumed.status_code == 200
    assert service.resumed == ["mmou_job_test"]
    assert trace.status_code == 200
    assert trace.json()["events"][0]["kind"] == "run_start"
