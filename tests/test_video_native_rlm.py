"""Focused tests for the native Gemini video RLM path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sanjaya import Agent
from sanjaya.prompts import PromptConfig
from sanjaya.tracing.tracer import Tracer
from sanjaya.tools.video.toolkit import VideoToolkit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "api"))


class DummyMediaLLM:
    def __init__(self, responses: list[Any], *, model: str):
        self._responses = list(responses)
        self.model = model
        self.vision_model = model
        self.calls: list[dict[str, Any]] = []
        self.last_usage = None
        self.last_call_metadata = None
        self.last_cost_usd = None

    def media_completion(self, *, prompt: str, media: list[dict[str, Any]], model: str | None = None) -> str:
        self.calls.append({
            "prompt": prompt,
            "media": media,
            "model": model or self.model,
        })
        return self._responses.pop(0)

    def media_completion_structured(
        self,
        *,
        prompt: str,
        media: list[dict[str, Any]],
        output_type: type[Any],
        model: str | None = None,
    ) -> Any:
        self.calls.append({
            "prompt": prompt,
            "media": media,
            "model": model or self.model,
            "output_type": output_type,
        })
        response = self._responses.pop(0)
        if isinstance(response, output_type):
            return response
        return output_type.model_validate(response)


def _write_artifact(path: str, payload: bytes = b"ok") -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return str(out)


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake video")
    return video


@pytest.fixture
def toolkit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_video: Path) -> VideoToolkit:
    monkeypatch.setattr("sanjaya.tools.video.toolkit.video_duration_seconds", lambda _: 90.0)
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit.get_video_info",
        lambda _: {
            "duration_s": 90.0,
            "width": 1920,
            "height": 1080,
            "codec": "h264",
            "fps": 29.97,
            "container": "mov,mp4,m4a,3gp,3g2,mj2",
            "file_size_mb": 12.34,
        },
    )
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit._extract_frame_impl",
        lambda video_path, at_s, output_path: _write_artifact(output_path, b"frame"),
    )

    tk = VideoToolkit(workspace_dir=str(tmp_path / "artifacts"))
    tk._inspect_llm_client = DummyMediaLLM(["frame result"], model="google-vertex:gemini-3.1-pro-preview")
    tk._llm_client = DummyMediaLLM(["submodel fallback"], model="google-vertex:gemini-3.1-flash-preview")
    tk._audio_llm_client = DummyMediaLLM(
        [{
            "transcript": "hello there",
            "audio_summary": "two people speaking",
            "salient_audio_events": ["speech", "brief pause"],
        }],
        model="google-vertex:gemini-3-flash-preview",
    )
    tk.setup({
        "video": str(fake_video),
        "question": "What is in the video?",
        "context": {"run_id": "test-run"},
        "modality": "balanced",
    })
    return tk


def test_video_toolkit_surface_includes_metadata(toolkit: VideoToolkit):
    tool_names = {tool.name for tool in toolkit.tools()}
    assert tool_names == {"get_video_info", "inspect_video", "analyze_audio"}


def test_get_video_info_returns_metadata(toolkit: VideoToolkit):
    get_info = next(tool.fn for tool in toolkit.tools() if tool.name == "get_video_info")

    result = get_info()

    assert result == {
        "duration_s": 90.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "fps": 29.97,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "file_size_mb": 12.34,
    }


def test_zero_length_video_inspection_becomes_single_frame(toolkit: VideoToolkit):
    inspect = next(tool.fn for tool in toolkit.tools() if tool.name == "inspect_video")

    result = inspect(start_s=12.0, end_s=12.0)

    assert "Queued frame attachment" in result
    assert toolkit._pending_root_media[0]["media"][0]["media_type"] == "image/jpeg"
    assert toolkit._llm_client.calls == []
    assert toolkit._inspect_llm_client.calls == []

    state = toolkit.get_state()
    assert state["recent_inspected_spans"][-1]["kind"] == "frame"
    assert state["single_frame_inspections"][-1]["start_s"] == 12.0
    assert state["pending_root_inspections"][-1]["start_s"] == 12.0


def test_audio_analysis_returns_structured_payload(toolkit: VideoToolkit):
    analyze = next(tool.fn for tool in toolkit.tools() if tool.name == "analyze_audio")

    result = analyze(start_s=15.0, end_s=30.0, prompt="Focus on speakers")

    assert result["transcript"] == "hello there"
    assert result["audio_summary"] == "two people speaking"
    assert result["salient_audio_events"] == ["speech", "brief pause"]

    llm = toolkit._audio_llm_client
    assert llm.calls[0]["media"][0]["media_type"] == "video/mp4"
    assert llm.calls[0]["output_type"].__name__ == "AudioAnalysisResult"
    assert llm.calls[0]["media"][0]["vendor_metadata"] == {
        "start_offset": "15.000s",
        "end_offset": "30.000s",
    }
    assert "fps" not in llm.calls[0]["media"][0]["vendor_metadata"]

    state = toolkit.get_state()
    assert state["recent_audio_spans"][-1]["start_s"] == 15.0
    assert state["recent_audio_spans"][-1]["end_s"] == 30.0


def test_media_trace_context_respects_delegated_depth(toolkit: VideoToolkit):
    tracer = Tracer(enabled=False, track_events=True)
    toolkit._tracer = tracer

    with toolkit._media_trace_context(
        kind="video",
        model="google-vertex:gemini-3-flash-preview",
        prompt="Check this slice",
        start_s=10.0,
        end_s=20.0,
        source="llm_query_batched",
        trace_depth=2,
    ) as trace:
        trace.record_response("found it")

    end_event = tracer.events[-1]
    assert end_event["kind"] == "sanjaya.video_inspection_end"
    assert end_event["source"] == "llm_query_batched"
    assert end_event["depth"] == 2


def test_span_validation_and_child_scoping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_video: Path):
    monkeypatch.setattr("sanjaya.tools.video.toolkit.video_duration_seconds", lambda _: 300.0)

    toolkit = VideoToolkit(workspace_dir=str(tmp_path / "artifacts"))
    toolkit.setup({
        "video": str(fake_video),
        "question": "test",
        "context": {"run_id": "test-run"},
    })

    with pytest.raises(ValueError, match="hard 120.0s limit"):
        toolkit.prepare_media_request(start_s=0.0, end_s=121.0, media_kind="video")

    with pytest.raises(ValueError, match="Zero-length spans"):
        toolkit.prepare_media_request(start_s=8.0, end_s=8.0, media_kind="audio")

    child = toolkit.spawn_child(active_span=(10.0, 20.0))
    with pytest.raises(ValueError, match="outside the active child span"):
        child.prepare_media_request(start_s=9.0, end_s=12.0, media_kind="video")

    request = child.prepare_media_request(start_s=10.0, end_s=20.0, media_kind="video")
    assert request["kind"] == "video"
    assert request["artifact_path"] == str(fake_video.resolve())
    assert request["media"][0]["path"] == str(fake_video.resolve())
    assert request["media"][0]["vendor_metadata"] == {
        "start_offset": "10.000s",
        "end_offset": "20.000s",
    }
    assert "fps" not in request["media"][0]["vendor_metadata"]


def test_agent_defaults_and_api_contract():
    agent = Agent(tracing=False)

    assert agent.model == "google-vertex:gemini-3.1-pro-preview"
    assert agent.sub_model == "google-vertex:gemini-3-flash-preview"
    assert agent.vision_model == "google-vertex:gemini-3.1-pro-preview"
    assert agent.audio_model == "google-vertex:gemini-3-flash-preview"
    assert agent.fallback_model is None

    from sanjaya_api.models import RunRequest

    schema = RunRequest.model_json_schema()
    assert schema["properties"]["max_depth"]["default"] == 1
    assert "subtitle_mode" not in schema.get("properties", {})
    assert "subtitle_api_model" not in schema.get("properties", {})


def test_agent_binds_video_inspection_to_root_model() -> None:
    toolkit = VideoToolkit()
    agent = Agent(
        model="root-model",
        sub_model="sub-model",
        vision_model="vision-model",
        tracing=False,
    )

    agent.use(toolkit)

    assert toolkit._inspect_llm_client is agent._orchestrator
    assert toolkit._llm_client is agent._sub_llm
    assert agent._orchestrator.vision_model == "vision-model"


def test_prompt_mentions_context_rot_and_batched_slice_docs():
    from sanjaya.core.prompts import _CORE_INSTRUCTIONS_RECURSIVE

    assert "avoid context rot" in _CORE_INSTRUCTIONS_RECURSIVE.lower()
    assert "promptless" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "llm_query_batched" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "rlm_query_batched" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "start_s" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "end_s" in _CORE_INSTRUCTIONS_RECURSIVE


def test_promptless_inspect_video_is_consumed_by_next_root_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    monkeypatch.setattr("sanjaya.tools.video.toolkit.video_duration_seconds", lambda _: 90.0)
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit.get_video_info",
        lambda _: {
            "duration_s": 90.0,
            "width": 1920,
            "height": 1080,
            "codec": "h264",
            "fps": 29.97,
            "container": "mov,mp4,m4a,3gp,3g2,mj2",
            "file_size_mb": 12.34,
        },
    )
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit._extract_frame_impl",
        lambda video_path, at_s, output_path: _write_artifact(output_path, b"frame"),
    )
    monkeypatch.setattr(
        "sanjaya.core.schema.classify_question_modality",
        lambda question, llm_client: "balanced",
    )

    completion_calls: list[str] = []
    completion_with_media_calls: list[dict[str, Any]] = []

    toolkit = VideoToolkit(workspace_dir=str(tmp_path / "artifacts"))
    agent = Agent(
        prompts=PromptConfig(
            answer_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        ),
        tracing=False,
        max_iterations=2,
    )
    agent.use(toolkit)

    def fake_completion(prompt: str, timeout: int = 300) -> str:
        completion_calls.append(prompt)
        return "```python\ninspect_video(start_s=12.0, end_s=12.0)\n```"

    def fake_completion_with_media(prompt: str, media: list[dict[str, Any]], timeout: int = 300) -> str:
        completion_with_media_calls.append({"prompt": prompt, "media": media})
        return '```python\ndone({"answer": "attached slice consumed"})\n```'

    monkeypatch.setattr(agent._orchestrator, "completion", fake_completion)
    monkeypatch.setattr(agent._orchestrator, "completion_with_media", fake_completion_with_media)

    answer = agent.ask(
        "What happens at 12 seconds?",
        context={"run_id": "promptless-inspect"},
        video=str(fake_video),
    )

    assert answer.text == "attached slice consumed"
    assert len(completion_calls) == 1
    assert len(completion_with_media_calls) == 1
    media_item = completion_with_media_calls[0]["media"][0]
    assert media_item["media_type"] == "image/jpeg"
    assert Path(media_item["path"]).name.endswith(".jpg")


def test_incomplete_video_run_persists_partial_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
):
    monkeypatch.setattr("sanjaya.tools.video.toolkit.video_duration_seconds", lambda _: 90.0)
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit.get_video_info",
        lambda _: {
            "duration_s": 90.0,
            "width": 1920,
            "height": 1080,
            "codec": "h264",
            "fps": 29.97,
            "container": "mov,mp4,m4a,3gp,3g2,mj2",
            "file_size_mb": 12.34,
        },
    )
    monkeypatch.setattr("sanjaya.agent.classify_question_modality", lambda question, llm_client: "balanced", raising=False)
    monkeypatch.setattr(
        "sanjaya.core.schema.classify_question_modality",
        lambda question, llm_client: "balanced",
    )
    monkeypatch.setattr(
        "sanjaya.core.schema.generate_answer_schema",
        lambda question, llm_client: {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    def fail_run_loop(**kwargs):
        kwargs["tracer"].emit("test.partial_trace", note="before failure")
        raise RuntimeError("loop exploded")

    monkeypatch.setattr("sanjaya.agent.run_loop", fail_run_loop)

    toolkit = VideoToolkit(workspace_dir=str(tmp_path / "artifacts"))
    agent = Agent(tracing=True)
    agent.use(toolkit)

    with pytest.raises(RuntimeError, match="loop exploded"):
        agent.ask(
            "What happened?",
            context={"run_id": "partial-trace-run"},
            video=str(fake_video),
        )

    run_dir = tmp_path / "artifacts" / "partial-trace-run"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))

    assert trace["status"] == "error"
    assert trace["error"] == "loop exploded"
    assert trace["wall_time_s"] is not None
    assert trace["events"]
    assert any(event["kind"] == "sanjaya.completion_start" for event in trace["events"])
    assert any(event["kind"] == "sanjaya.completion_end" for event in trace["events"])
    assert any(event["kind"] == "test.partial_trace" for event in manifest["trace_events"])


def test_recursive_subcall_trace_event_is_persisted() -> None:
    from sanjaya.tracing.tracer import Tracer
    from sanjaya_api.trace_events import normalize_trace_event

    tracer = Tracer(enabled=False, track_events=True)

    with tracer.subcall(
        depth=1,
        prompt="Inspect the decisive moment in the clip.",
        child_model="google-vertex/gemini-3.1-pro-preview",
        start_s=12.0,
        end_s=18.5,
    ) as trace:
        trace.record_response("The player raises both arms after the shot.")
        trace.record(status="complete", iterations_used=2)

    raw_events = tracer.dump_events()
    assert [event["kind"] for event in raw_events][-2:] == [
        "sanjaya.rlm_subcall_start",
        "sanjaya.rlm_subcall_end",
    ]

    kind, _, payload = normalize_trace_event(raw_events[-1])
    assert kind == "subcall"
    assert payload["depth"] == 1
    assert payload["iterations_used"] == 2
    assert payload["prompt_content"] == "Inspect the decisive moment in the clip."
    assert payload["response_preview"] == "The player raises both arms after the shot."
