from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import sanjaya.benchmarks.mmou_adapter as mmou_adapter
from sanjaya.answer import Answer
from sanjaya.benchmarks.mmou_adapter import SanjayaMMOUAdapter


class FakeGenerationResponse:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeRequest(SimpleNamespace):
    def cache_key(self, adapter_name: str) -> str:
        self.cache_adapter_name = adapter_name
        return "stable-cache-key"


def test_mmou_prompt_sanitizer_removes_summary_and_benchmark_fields() -> None:
    raw = (
        "Whole Video Summary:\n"
        "A spoiler summary that should not be provided.\n\n"
        "Use the summary as high-level context for the full video, but answer based on the provided video clip.\n\n"
        "Answer the multiple-choice question about the video.\n"
        "Return only JSON with one key: {\"answer\": \"A\"}.\n\n"
        "Question ID: q-1\n"
        "Domain: Sports\n"
        "Question: What happens next?\n"
        "Evidence Window: 00:10 to 00:20\n"
        "Options:\nA. Alpha\nB. Beta\n"
    )

    prompt = mmou_adapter._sanitize_mmou_prompt(raw)

    assert "Question: What happens next?" in prompt
    assert "Options:" in prompt
    assert "Whole Video Summary" not in prompt
    assert "spoiler summary" not in prompt
    assert "Question ID" not in prompt
    assert "Domain:" not in prompt
    assert "Evidence Window" not in prompt
    assert "MMOU" not in prompt
    assert "benchmark" not in prompt.lower()


def test_mmou_adapter_wraps_sanjaya_answer_and_cleans_temp_artifacts(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    instances: list[Any] = []

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            instances.append(self)

        def use(self, toolkit: Any) -> "FakeAgent":
            self.toolkit = toolkit
            return self

        def ask(self, question: str, *, context: Any = None, video: str | None = None, **kwargs: Any) -> Answer:
            self.ask_call = {
                "question": question,
                "context": context,
                "video": video,
            }
            self.workspace_dir = Path(self.toolkit.workspace_dir)
            assert self.workspace_dir.exists()
            return Answer(
                question=question,
                text="The correct option is c.",
                data={"answer": "c"},
                evidence=[],
                iterations=3,
                cost_usd=0.42,
                input_tokens=100,
                output_tokens=25,
                wall_time_s=12.5,
            )

    monkeypatch.setattr(mmou_adapter, "_generation_response_class", lambda: FakeGenerationResponse)
    request = FakeRequest(
        model="sanjaya-rlm",
        prompt=(
            "Answer the multiple-choice question about the video.\n"
            "Return only JSON with one key: {\"answer\": \"A\"}.\n\n"
            "Question ID: q-1\n"
            "Domain: Sports\n"
            "Question: What happens next?\n"
            "Evidence Window: 00:10 to 00:20\n"
            "Options:\nA. Alpha\nB. Beta\n"
        ),
        media=[SimpleNamespace(path=video_path)],
        metadata={"question_id": "q-1"},
    )

    response = SanjayaMMOUAdapter(
        root_model="root",
        sub_model="flash",
        recursive_model="pro-child",
        vision_model="vision",
        audio_model="audio",
        agent_factory=FakeAgent,
    ).generate(request)

    assert json.loads(response.text) == {"answer": "C"}
    assert response.parsed_json == {"answer": "C"}
    assert response.model == "sanjaya-rlm"
    assert response.cache_key == "stable-cache-key"
    assert response.cached is False
    assert response.usage == {
        "input_tokens": 100,
        "output_tokens": 25,
        "cost_usd": 0.42,
        "wall_time_s": 12.5,
        "iterations": 3,
        "evidence_count": 0,
    }

    agent = instances[0]
    assert agent.kwargs["sub_model"] == "flash"
    assert agent.kwargs["recursive_model"] == "pro-child"
    assert agent.kwargs["critic_model"] is None
    assert agent.kwargs["caption_model"] is None
    assert "Question: What happens next?" in agent.ask_call["question"]
    assert "Options:" in agent.ask_call["question"]
    assert "Question ID" not in agent.ask_call["question"]
    assert "Domain:" not in agent.ask_call["question"]
    assert "Evidence Window" not in agent.ask_call["question"]
    assert "MMOU" not in agent.ask_call["question"]
    assert "benchmark" not in agent.ask_call["question"].lower()
    assert agent.ask_call["context"] is None
    assert agent.ask_call["video"] == str(video_path)
    assert request.cache_adapter_name == "sanjaya-rlm"
    assert not agent.workspace_dir.exists()
