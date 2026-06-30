"""Focused tests for the native Gemini video RLM path."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from sanjaya import Agent
from sanjaya.prompts import PromptConfig
from sanjaya.tools.video.toolkit import VideoToolkit
from sanjaya.tracing.tracer import Tracer

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
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit._extract_zoomed_frame_impl",
        lambda video_path, at_s, output_path, zoom_box, source_width, source_height: _write_artifact(
            output_path,
            b"zoomed frame",
        ),
    )
    monkeypatch.setattr(
        "sanjaya.tools.video.toolkit._extract_zoomed_clip_impl",
        lambda video_path, start_s, end_s, output_path, zoom_box, source_width, source_height: _write_artifact(
            output_path,
            b"zoomed clip",
        ),
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


def test_inspect_video_supports_coordinate_zoom(toolkit: VideoToolkit):
    inspect = next(tool.fn for tool in toolkit.tools() if tool.name == "inspect_video")

    result = inspect(start_s=12.0, end_s=12.0, zoom_box=(0, 0, 500, 500))

    assert "with zoom" in result
    queued = toolkit._pending_root_media[-1]
    media_item = queued["media"][0]
    assert media_item["media_type"] == "image/jpeg"
    assert "zoom_" in Path(media_item["path"]).name
    assert queued["zoom_box"] == (0.0, 0.0, 500.0, 500.0)
    assert queued["effective_zoom_box"][:2] == (0.0, 0.0)
    assert queued["effective_zoom_box"][2] == pytest.approx(888.8888888889)
    assert queued["effective_zoom_box"][3] == 500.0

    state = toolkit.get_state()
    assert state["recent_inspected_spans"][-1]["zoom_box"] == [0.0, 0.0, 500.0, 500.0]
    assert state["uploaded_file_status"]["cached_zoomed_media"] == 1


def test_inspect_video_rejects_invalid_zoom_box(toolkit: VideoToolkit):
    inspect = next(tool.fn for tool in toolkit.tools() if tool.name == "inspect_video")

    with pytest.raises(ValueError, match="x2 > x1"):
        inspect(start_s=12.0, end_s=12.0, zoom_box=(100, 100, 100, 200))


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


def test_zoomed_llm_query_sends_cropped_media(toolkit: VideoToolkit):
    from sanjaya.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_toolkit(toolkit)
    agent = Agent(tracing=False)

    result = agent._run_sub_llm_query(
        registry=registry,
        repl=None,
        prompt="Inspect the lower half.",
        start_s=10.0,
        end_s=20.0,
        zoom_box=(0, 500, 1000, 1000),
        client=toolkit._llm_client,
    )

    assert result == "submodel fallback"
    media_item = toolkit._llm_client.calls[0]["media"][0]
    assert media_item["media_type"] == "video/mp4"
    assert "vendor_metadata" not in media_item
    assert "zoom_" in Path(media_item["path"]).name
    assert toolkit.get_state()["recent_inspected_spans"][-1]["zoom_box"] == [0.0, 500.0, 1000.0, 1000.0]


def test_child_zoom_scope_composes_with_nested_zoom(toolkit: VideoToolkit):
    child = toolkit.spawn_child(active_span=(10.0, 20.0), active_zoom_box=(0, 0, 500, 500))

    assert child.get_state()["active_zoom_box"] == list(toolkit.effective_zoom_box((0, 0, 500, 500)))

    request = child.prepare_media_request(
        start_s=12.0,
        end_s=12.0,
        media_kind="video",
        zoom_box=(500, 0, 1000, 1000),
    )

    assert request["kind"] == "frame"
    assert request["zoom_box"] == (500.0, 0.0, 1000.0, 1000.0)
    assert request["effective_zoom_box"][0] > 0.0
    assert request["effective_zoom_box"][2] <= 1000.0
    assert "zoom_" in Path(request["artifact_path"]).name


def test_agent_defaults_and_api_contract():
    agent = Agent(tracing=False)

    assert agent.model == "google-vertex:gemini-3.1-pro-preview"
    assert agent.sub_model == "google-vertex:gemini-3-flash-preview"
    assert agent.recursive_model == "google-vertex:gemini-3-flash-preview"
    assert agent.vision_model == "google-vertex:gemini-3.1-pro-preview"
    assert agent.audio_model == "google-vertex:gemini-3-flash-preview"
    assert agent.fallback_model is None
    assert agent.auto_transcribe_video is True
    assert agent.transcription_model == "whisper-1"
    assert agent.transcription_language is None

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


def test_recursive_model_controls_child_rlm_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    from sanjaya.core.loop import LoopResult

    agent = Agent(
        model="root-model",
        sub_model="direct-sub-model",
        recursive_model="child-rlm-model",
        vision_model=None,
        audio_model=None,
        caption_model=None,
        critic_model=None,
        max_depth=2,
        tracing=False,
    )
    assert agent._sub_llm.model == "direct-sub-model"

    captured: dict[str, Any] = {}

    class FakeLLMClient:
        def __init__(
            self,
            model: str,
            vision_model: str | None = None,
            fallback_model: str | None = None,
            name: str = "llm",
        ):
            self.model = model
            self.vision_model = vision_model or model
            self.fallback_model = fallback_model
            self.name = name

    def fake_run_loop(**kwargs: Any) -> LoopResult:
        captured["orchestrator_model"] = kwargs["orchestrator"].model
        captured["orchestrator_vision_model"] = kwargs["orchestrator"].vision_model
        return LoopResult(
            raw_answer={"answer": "child ok"},
            iterations_used=1,
            messages=[],
            budget=kwargs["budget"],
            wall_time_s=0.1,
        )

    monkeypatch.setattr("sanjaya.agent.LLMClient", FakeLLMClient)
    monkeypatch.setattr("sanjaya.agent.run_loop", fake_run_loop)

    result = agent._subcall(
        "Inspect the child task.",
        depth=1,
        parent_run_registry=agent._build_runtime_registry(),
    )

    assert result == "child ok"
    assert captured == {
        "orchestrator_model": "child-rlm-model",
        "orchestrator_vision_model": "child-rlm-model",
    }


def test_prompt_mentions_context_rot_and_batched_slice_docs():
    from sanjaya.core.prompts import _CORE_INSTRUCTIONS, _CORE_INSTRUCTIONS_RECURSIVE

    assert "avoid context rot" in _CORE_INSTRUCTIONS_RECURSIVE.lower()
    assert "promptless" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "llm_query_batched" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "rlm_query_batched" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "start_s" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "end_s" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "zoom_box" in _CORE_INSTRUCTIONS
    assert "0-1000 coordinates" in _CORE_INSTRUCTIONS_RECURSIVE
    assert "the question may not be answerable" in _CORE_INSTRUCTIONS
    assert "the question may not be answerable" in _CORE_INSTRUCTIONS_RECURSIVE


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
        auto_transcribe_video=False,
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
    agent = Agent(tracing=True, auto_transcribe_video=False)
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


def test_repl_exposes_transcript_as_injected_variable() -> None:
    from sanjaya.core.repl import AgentREPL
    from sanjaya.tools.registry import ToolRegistry

    transcript = {
        "text": "hello there",
        "segments": [{"index": 0, "start_s": 1.0, "end_s": 2.0, "speaker": "speaker_0", "text": "hello there"}],
        "metadata": {"segment_count": 1},
    }
    repl = AgentREPL(registry=ToolRegistry(), inputs={"transcript": transcript})

    first = repl.execute("print(transcript['segments'][0]['text'])")
    repl.execute("transcript = {'segments': []}")
    second = repl.execute("print(transcript['segments'][0]['text'])")

    assert first.stdout == "hello there\n"
    assert second.stdout == "hello there\n"


def test_agent_injects_transcript_without_putting_text_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.core.loop import LoopResult

    monkeypatch.setattr("sanjaya.tools.video.toolkit.video_duration_seconds", lambda _: 30.0)
    monkeypatch.setattr(
        "sanjaya.core.schema.classify_question_modality",
        lambda question, llm_client: "balanced",
    )

    transcript = {
        "text": "secret spoken words",
        "segments": [
            {
                "index": 0,
                "start_s": 3.0,
                "end_s": 5.0,
                "speaker": "speaker_0",
                "text": "secret spoken words",
            }
        ],
        "metadata": {"segment_count": 1, "source": "test"},
    }
    captured: dict[str, Any] = {}

    def fake_run_loop(**kwargs: Any) -> LoopResult:
        captured["repl_inputs"] = kwargs["repl"].inputs
        captured["system_prompt"] = kwargs["system_prompt"]
        return LoopResult(
            raw_answer={"answer": "ok"},
            iterations_used=1,
            messages=[],
            budget=kwargs["budget"],
            wall_time_s=0.1,
        )

    agent = Agent(
        prompts=PromptConfig(
            answer_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        ),
        tracing=False,
    )
    sanitized_transcript = agent._repl_transcript(transcript)
    agent._prepare_video_transcript = lambda video, subtitle: sanitized_transcript  # type: ignore[method-assign]
    agent.use(VideoToolkit(workspace_dir=str(tmp_path / "artifacts")))
    monkeypatch.setattr("sanjaya.agent.run_loop", fake_run_loop)

    answer = agent.ask("What is said?", context={"run_id": "transcript-test"}, video=str(fake_video))

    assert answer.text == "ok"
    assert captured["repl_inputs"]["transcript"] == sanitized_transcript
    assert "source" not in captured["repl_inputs"]["transcript"]["metadata"]
    assert "A `transcript` dict is available" in captured["system_prompt"]
    assert "secret spoken words" not in captured["system_prompt"]


def test_recursive_child_receives_sliced_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.core.loop import LoopResult

    monkeypatch.setattr("sanjaya.tools.video.toolkit.video_duration_seconds", lambda _: 60.0)

    transcript = {
        "text": "inside outside",
        "segments": [
            {"index": 0, "start_s": 12.0, "end_s": 14.0, "speaker": "speaker_0", "text": "inside"},
            {"index": 1, "start_s": 25.0, "end_s": 26.0, "speaker": "speaker_1", "text": "outside"},
        ],
        "metadata": {"segment_count": 2},
    }
    captured: dict[str, Any] = {}

    def fake_run_loop(**kwargs: Any) -> LoopResult:
        captured["transcript"] = kwargs["repl"].inputs["transcript"]
        captured["system_prompt"] = kwargs["system_prompt"]
        return LoopResult(
            raw_answer={"answer": "child ok"},
            iterations_used=1,
            messages=[],
            budget=kwargs["budget"],
            wall_time_s=0.1,
        )

    toolkit = VideoToolkit(workspace_dir=str(tmp_path / "artifacts"))
    toolkit.setup({"video": str(fake_video), "question": "test", "context": {"run_id": "child-slice"}})
    agent = Agent(max_depth=2, tracing=False)
    agent.use(toolkit)
    monkeypatch.setattr("sanjaya.agent.run_loop", fake_run_loop)

    result = agent._subcall(
        "Inspect the child task.",
        depth=1,
        parent_run_registry=agent._build_runtime_registry(),
        transcript=transcript,
        active_span=(10.0, 20.0),
    )

    assert result == "child ok"
    assert captured["transcript"]["text"] == "inside"
    assert len(captured["transcript"]["segments"]) == 1
    assert captured["transcript"]["metadata"]["active_video_span"] == {"start_s": 10.0, "end_s": 20.0}
    assert "A `transcript` dict is available" in captured["system_prompt"]


def test_openai_whisper_transcription_request_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.tools.video import transcription

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    calls: list[dict[str, Any]] = []

    class FakeTranscriptions:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({key: value for key, value in kwargs.items() if key != "file"})
            return {
                "segments": [
                    {"start": 1.25, "end": 2.5, "text": "hello"},
                ]
            }

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.audio = types.SimpleNamespace(
                transcriptions=FakeTranscriptions(),
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(transcription, "_audio_upload_chunks", lambda src, tmp: [(audio_path, 0.0)])

    output_path = tmp_path / "transcript.json"
    generated = transcription.transcribe_with_openai_api(
        video_path=str(fake_video),
        output_path=str(output_path),
    )
    payload = json.loads(Path(generated).read_text(encoding="utf-8"))

    assert calls == [{
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }]
    assert "chunking_strategy" not in calls[0]
    assert payload["text"] == "hello"
    assert payload["segments"] == [{
        "start_s": 1.25,
        "end_s": 2.5,
        "text": "hello",
        "index": 0,
    }]
    assert payload["metadata"]["speaker_label_scope"] == "none"
    assert payload["metadata"]["language"] == "auto"


def test_openai_transcription_includes_explicit_language_only_when_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.tools.video import transcription

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    calls: list[dict[str, Any]] = []

    class FakeTranscriptions:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({key: value for key, value in kwargs.items() if key != "file"})
            return {"segments": []}

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.audio = types.SimpleNamespace(transcriptions=FakeTranscriptions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(transcription, "_audio_upload_chunks", lambda src, tmp: [(audio_path, 0.0)])

    generated = transcription.transcribe_with_openai_api(
        video_path=str(fake_video),
        output_path=str(tmp_path / "transcript.json"),
        language="ja",
    )
    payload = json.loads(Path(generated).read_text(encoding="utf-8"))

    assert calls[0]["language"] == "ja"
    assert payload["metadata"]["language"] == "ja"


def test_openai_transcription_allows_empty_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.tools.video import transcription

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    sleeps: list[float] = []

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return {"segments": []}

    fake_transcriptions = FakeTranscriptions()

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.audio = types.SimpleNamespace(transcriptions=fake_transcriptions)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(transcription, "_audio_upload_chunks", lambda src, tmp: [(audio_path, 0.0)])
    monkeypatch.setattr(transcription.time, "sleep", lambda seconds: sleeps.append(seconds))

    generated = transcription.transcribe_with_openai_api(
        video_path=str(fake_video),
        output_path=str(tmp_path / "transcript.json"),
    )
    payload = json.loads(Path(generated).read_text(encoding="utf-8"))

    assert sleeps == []
    assert fake_transcriptions.calls == 1
    assert payload["text"] == ""
    assert payload["segments"] == []
    assert payload["metadata"]["segment_count"] == 0


def test_openai_transcription_retries_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.tools.video import transcription

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    sleeps: list[float] = []

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary API failure")
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "ready"}]}

    fake_transcriptions = FakeTranscriptions()

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.audio = types.SimpleNamespace(transcriptions=fake_transcriptions)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(transcription, "_audio_upload_chunks", lambda src, tmp: [(audio_path, 0.0)])
    monkeypatch.setattr(transcription.time, "sleep", lambda seconds: sleeps.append(seconds))

    generated = transcription.transcribe_with_openai_api(
        video_path=str(fake_video),
        output_path=str(tmp_path / "transcript.json"),
    )
    payload = json.loads(Path(generated).read_text(encoding="utf-8"))

    assert sleeps == [1.0, 2.0]
    assert fake_transcriptions.calls == 3
    assert payload["text"] == "ready"


def test_openai_transcription_retries_failures_for_one_minute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
) -> None:
    from sanjaya.tools.video import transcription

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    sleeps: list[float] = []

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise RuntimeError("temporary API failure")

    fake_transcriptions = FakeTranscriptions()

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.audio = types.SimpleNamespace(transcriptions=fake_transcriptions)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(transcription, "_audio_upload_chunks", lambda src, tmp: [(audio_path, 0.0)])
    monkeypatch.setattr(transcription.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RuntimeError, match="temporary API failure"):
        transcription.transcribe_with_openai_api(
            video_path=str(fake_video),
            output_path=str(tmp_path / "transcript.json"),
        )

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 29.0]
    assert sum(sleeps) == 60.0
    assert fake_transcriptions.calls == 7


def test_recursive_subcall_trace_event_is_persisted() -> None:
    from sanjaya_api.trace_events import normalize_trace_event

    from sanjaya.tracing.tracer import Tracer

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
