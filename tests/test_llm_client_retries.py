from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from sanjaya.llm.client import EmptyModelResponseError, LLMClient


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
