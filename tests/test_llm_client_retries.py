from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from sanjaya.llm.client import EmptyModelResponseError, LLMClient, _compute_cost


class StructuredAnswer(BaseModel):
    answer: str


@pytest.fixture
def dummy_result() -> SimpleNamespace:
    return SimpleNamespace(response=None)


def test_rate_limit_errors_retry_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
    dummy_result: SimpleNamespace,
) -> None:
    client = LLMClient(model="google-vertex:gemini-3.1-pro-preview")
    sleeps: list[float] = []
    attempts = {"count": 0}

    def fake_run_agent(model: str, payload: str, **kwargs: object) -> tuple[str, SimpleNamespace]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ModelHTTPError(
                429,
                "gemini-3.1-pro-preview",
                {"error": {"status": "RESOURCE_EXHAUSTED"}},
            )
        return "final answer", dummy_result

    monkeypatch.setattr(client, "_run_agent", fake_run_agent)
    monkeypatch.setattr("sanjaya.llm.client.time.sleep", lambda delay: sleeps.append(delay))

    result = client.completion("hello")

    assert result == "final answer"
    assert sleeps == [4.0, 8.0]


def test_empty_text_responses_retry_before_succeeding(
    monkeypatch: pytest.MonkeyPatch,
    dummy_result: SimpleNamespace,
) -> None:
    client = LLMClient(model="google-vertex:gemini-3.1-pro-preview")
    sleeps: list[float] = []
    responses = iter([
        ("   ", dummy_result),
        ("recovered", dummy_result),
    ])

    monkeypatch.setattr(client, "_run_agent", lambda model, payload, **kwargs: next(responses))
    monkeypatch.setattr("sanjaya.llm.client.time.sleep", lambda delay: sleeps.append(delay))

    result = client.completion("hello")

    assert result == "recovered"
    assert sleeps == [1.0]


def test_structured_output_validation_retries_with_sdk_output_retries(
    monkeypatch: pytest.MonkeyPatch,
    dummy_result: SimpleNamespace,
) -> None:
    client = LLMClient(model="google-vertex:gemini-3-flash-preview")
    sleeps: list[float] = []
    run_kwargs: list[dict[str, object]] = []

    def fake_run_agent(model: str, payload: object, **kwargs: object) -> tuple[StructuredAnswer, SimpleNamespace]:
        run_kwargs.append(dict(kwargs))
        if len(run_kwargs) == 1:
            raise UnexpectedModelBehavior("Exceeded maximum retries (1) for output validation")
        return StructuredAnswer(answer="ok"), dummy_result

    monkeypatch.setattr(client, "_run_agent", fake_run_agent)
    monkeypatch.setattr("sanjaya.llm.client.time.sleep", lambda delay: sleeps.append(delay))

    result = client._call_structured(
        "google-vertex:gemini-3-flash-preview",
        "hello",
        output_type=StructuredAnswer,
    )

    assert result.answer == "ok"
    assert run_kwargs[0]["output_retries"] == 3
    assert sleeps == [1.5]


def test_exhausted_empty_responses_raise_non_empty_error(
    monkeypatch: pytest.MonkeyPatch,
    dummy_result: SimpleNamespace,
) -> None:
    client = LLMClient(model="google-vertex:gemini-3.1-pro-preview")

    monkeypatch.setattr(client, "_run_agent", lambda model, payload, **kwargs: ("", dummy_result))
    monkeypatch.setattr("sanjaya.llm.client.time.sleep", lambda delay: None)

    with pytest.raises(EmptyModelResponseError, match="Model returned an empty response"):
        client.completion("hello")


def test_gemini3_vertex_cost_uses_current_standard_rates() -> None:
    cost = _compute_cost(
        "google-vertex:gemini-3.1-pro-preview",
        input_tokens=46_095,
        output_tokens=439,
    )

    assert cost == pytest.approx(0.097458, abs=0.000001)


def test_gemini3_flash_audio_input_uses_audio_rate_when_available() -> None:
    cost = _compute_cost(
        "google-vertex:gemini-3-flash-preview",
        input_tokens=1_500,
        output_tokens=100,
        input_modality_tokens={"audio": 1_000, "text": 500},
    )

    assert cost == pytest.approx(0.00155, abs=0.000001)


def test_usage_capture_understands_google_usage_metadata() -> None:
    client = LLMClient(model="google-vertex:gemini-3.1-pro-preview")
    result = SimpleNamespace(
        usage=lambda: SimpleNamespace(
            prompt_token_count=46_095,
            candidates_token_count=11,
            thoughts_token_count=428,
            total_token_count=46_534,
            prompt_tokens_details=[
                {"modality": "AUDIO", "token_count": 12_850},
                {"modality": "TEXT", "token_count": 349},
                {"modality": "VIDEO", "token_count": 32_896},
            ],
        ),
        response=SimpleNamespace(model_name="gemini-3.1-pro-preview"),
    )

    usage = client._capture_usage(result)
    cost = client._extract_cost_usd(result)

    assert usage.input_tokens == 46_095
    assert usage.output_tokens == 439
    assert usage.reasoning_tokens == 428
    assert usage.total_tokens == 46_534
    assert usage.input_modality_tokens == {"audio": 12850, "text": 349, "video": 32896}
    assert cost == pytest.approx(0.097458, abs=0.000001)
