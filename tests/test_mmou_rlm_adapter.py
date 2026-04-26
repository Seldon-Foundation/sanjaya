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
        prompt="Answer this MMOU question.",
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
    assert agent.ask_call["question"] == "Answer this MMOU question."
    assert agent.ask_call["context"]["question_id"] == "q-1"
    assert agent.ask_call["video"] == str(video_path)
    assert request.cache_adapter_name == "sanjaya-rlm"
    assert not agent.workspace_dir.exists()
