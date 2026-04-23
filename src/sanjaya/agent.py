"""Agent — the single entry point for sanjaya.

Replaces both RLM_REPL and VideoRLM_REPL with a unified, extensible API.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import OpenAIProvider

from .answer import Answer, Evidence
from .core.budget import BudgetTracker
from .core.loop import LoopConfig, LoopResult, _model_label, run_loop
from .core.prompts import build_system_prompt
from .core.repl import AgentREPL
from .llm.client import LLMClient, ModelSpec
from .model_defaults import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_CAPTION_MODEL,
    DEFAULT_CRITIC_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_ROOT_MODEL,
    DEFAULT_SUB_MODEL,
    DEFAULT_VISION_MODEL,
)
from .prompts import PromptConfig
from .tools.base import Tool, Toolkit
from .tools.builtins import (
    make_context_tool,
    make_done_tool,
    make_get_state_tool,
    make_llm_query_batched_tool,
    make_llm_query_tool,
    make_rlm_query_batched_tool,
    make_rlm_query_tool,
)
from .tools.registry import ToolRegistry
from .tracing import Tracer


def _resolve_model(
    spec: ModelSpec,
    provider: Provider | None,
    primary: ModelSpec | None = None,
) -> ModelSpec:
    """Resolve a model spec into a concrete Model when possible.

    Resolution order:
    1. Already a Model object — return as-is.
    2. A string + explicit ``provider`` — build an ``OpenAIResponsesModel``
       with that provider (the Responses API is OpenAI's preferred API).
    3. A string + a ``primary`` Model object — reuse the primary's provider
       to build a sibling of the same Model class.
    4. A plain provider-prefixed string like ``"openrouter:openai/gpt-4o"`` —
       return as-is and let pydantic-ai resolve it from env vars.
    """
    if isinstance(spec, Model):
        return spec

    # Moondream specs are handled directly by LLMClient — don't resolve
    from .llm.moondream import is_moondream_spec
    if is_moondream_spec(spec):
        return spec

    # Strip provider prefix for model name (e.g. "openrouter:openai/gpt-4.1-mini" → "openai/gpt-4.1-mini")
    model_name = spec.split(":", 1)[1] if ":" in spec else spec

    # Explicit provider takes precedence.
    # Use Responses API for direct OpenAI, Chat Completions for everything else
    # (OpenRouter, custom endpoints don't support Responses API).
    if provider is not None:
        try:
            model_cls = OpenAIResponsesModel if isinstance(provider, OpenAIProvider) else OpenAIChatModel
            return model_cls(model_name, provider=provider)
        except Exception:
            return spec

    # Inherit from primary model's provider
    if primary is not None and isinstance(primary, Model):
        inherited = getattr(primary, "_provider", None)
        if inherited is not None:
            try:
                return type(primary)(model_name, provider=inherited)
            except Exception:
                pass

    return spec


class Agent:
    """RLM agent that solves problems by writing code in a sandboxed REPL."""

    def __init__(
        self,
        model: ModelSpec = DEFAULT_ROOT_MODEL,
        sub_model: ModelSpec = DEFAULT_SUB_MODEL,
        vision_model: ModelSpec | None = DEFAULT_VISION_MODEL,
        audio_model: ModelSpec | None = DEFAULT_AUDIO_MODEL,
        caption_model: ModelSpec | None = DEFAULT_CAPTION_MODEL,
        fallback_model: ModelSpec | None = DEFAULT_FALLBACK_MODEL,
        critic_model: ModelSpec | None = DEFAULT_CRITIC_MODEL,
        *,
        prompts: PromptConfig | None = None,
        provider: Provider | None = None,
        max_iterations: int = 8,
        max_depth: int = 1,
        max_budget_usd: float | None = None,
        max_timeout_s: float | None = None,
        compaction_threshold: float = 0.85,
        critic_threshold: int = 70,
        tracing: bool = True,
    ):
        # Resolve all model specs through the provider chain.
        # The primary model is resolved first so siblings can inherit from it.
        model = _resolve_model(model, provider)
        sub_model = _resolve_model(sub_model, provider, primary=model)
        if vision_model is not None:
            vision_model = _resolve_model(vision_model, provider, primary=model)
        if audio_model is not None:
            audio_model = _resolve_model(audio_model, provider, primary=model)
        if fallback_model is not None:
            fallback_model = _resolve_model(fallback_model, provider, primary=model)

        self._prompts = prompts or PromptConfig()
        self.model = model
        self.sub_model = sub_model
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.caption_model = caption_model
        self.fallback_model = fallback_model
        self.max_iterations = max_iterations
        self._max_depth = max_depth
        self._depth = 0  # always 0 for user-created agents
        self.max_budget_usd = max_budget_usd
        self.max_timeout_s = max_timeout_s
        self.compaction_threshold = compaction_threshold
        self.critic_threshold = critic_threshold

        # LLM clients
        self._orchestrator = LLMClient(
            model=model,
            fallback_model=fallback_model,
            name="root_llm",
        )
        self._sub_llm = LLMClient(
            model=sub_model,
            vision_model=vision_model or sub_model,
            fallback_model=fallback_model,
            name="sub_llm",
        )
        self._critic = LLMClient(model=critic_model, name="critic") if critic_model else None
        self._audio_llm = LLMClient(
            model=audio_model,
            vision_model=audio_model,
            fallback_model=None,
            name="audio_llm",
        ) if audio_model else None

        # Captioner (separate from sub_llm — used only by caption_frames)
        self._captioner: Any = None
        if caption_model is not None:
            from .llm.moondream import MOONDREAM_STATION_BASE, MoondreamVisionClient, is_moondream_spec
            if is_moondream_spec(caption_model):
                spec = str(caption_model)
                use_station = spec.startswith("moondream-station:")
                model_id = spec.split(":", 1)[1] if ":" in spec else "moondream3-preview"
                try:
                    self._captioner = MoondreamVisionClient(
                        model=model_id,
                        base_url=MOONDREAM_STATION_BASE if use_station else None,
                    )
                except Exception:
                    pass

        # Tool registry
        self._registry = ToolRegistry()

        # Tracing
        self._tracer = Tracer(enabled=tracing, track_events=True)

        # Budget tracking (cumulative across ask() calls)
        self._budget = BudgetTracker(
            max_budget_usd=max_budget_usd,
            max_timeout_s=max_timeout_s,
        )

        # State
        self._last_answer: Answer | None = None

    def use(self, *tools_or_toolkits: Tool | Toolkit) -> "Agent":
        """Register tools or toolkits. Chainable."""
        for item in tools_or_toolkits:
            if isinstance(item, Toolkit):
                # Inject LLM client and tracer for vision-capable toolkits
                if hasattr(item, "_llm_client"):
                    item._llm_client = self._sub_llm
                if hasattr(item, "_tracer"):
                    item._tracer = self._tracer
                if hasattr(item, "_budget"):
                    item._budget = self._budget
                if hasattr(item, "_audio_llm_client"):
                    item._audio_llm_client = self._audio_llm
                if hasattr(item, "_captioner") and self._captioner is not None:
                    item._captioner = self._captioner
                item._prompt_config = self._prompts
                self._registry.register_toolkit(item)
            elif isinstance(item, Tool):
                self._registry.register(item)
            else:
                raise TypeError(f"Expected Tool or Toolkit, got {type(item).__name__}")
        return self

    def _build_runtime_registry(
        self,
        *,
        toolkits: list[Toolkit] | None = None,
        standalone_tools: list[Tool] | None = None,
    ) -> ToolRegistry:
        registry = ToolRegistry()
        active_toolkits = list(toolkits or self._registry.toolkits)
        toolkit_tool_names: set[str] = set()

        for toolkit in active_toolkits:
            registry.register_toolkit(toolkit)
            toolkit_tool_names.update(t.name for t in toolkit.tools())

        source_tools = standalone_tools if standalone_tools is not None else self._registry.all_tools()
        for tool in source_tools:
            if tool.name not in toolkit_tool_names:
                registry.register(tool)

        return registry

    def _make_sub_llm_client(self, *, name: str) -> LLMClient:
        client = LLMClient(
            model=self.sub_model,
            vision_model=self.vision_model or self.sub_model,
            fallback_model=self.fallback_model,
            name=name,
        )
        client._media_binary_cache = self._sub_llm._media_binary_cache
        return client

    def _find_video_toolkit(self, registry: ToolRegistry) -> Any | None:
        try:
            from .tools.video import VideoToolkit

            for toolkit in registry.toolkits:
                if isinstance(toolkit, VideoToolkit):
                    return toolkit
        except ImportError:
            return None
        return None

    def _spawn_child_toolkits(
        self,
        *,
        parent_registry: ToolRegistry,
        active_span: tuple[float, float] | None,
        depth: int,
    ) -> list[Toolkit]:
        child_toolkits: list[Toolkit] = []
        for toolkit in parent_registry.toolkits:
            spawn_child = getattr(toolkit, "spawn_child", None)
            if callable(spawn_child):
                try:
                    child_toolkits.append(spawn_child(active_span=active_span, trace_depth=depth))
                    continue
                except TypeError:
                    child_toolkits.append(spawn_child())
                    continue
            child_toolkits.append(toolkit)
        return child_toolkits

    def _record_client_usage(
        self,
        *,
        client: LLMClient,
        budget: BudgetTracker,
        default_model: str,
        trace: Any,
    ) -> None:
        usage = client.last_usage
        if usage:
            cost = client.last_cost_usd or 0.0
            budget.record(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                model=default_model,
            )
            trace.record(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=cost,
            )

        metadata = client.last_call_metadata
        if metadata:
            trace.record(
                model_used=metadata.model_used,
                provider=metadata.provider,
                duration_seconds=metadata.duration_seconds,
                fallback_used=metadata.fallback_used,
                cost_usd=metadata.cost_usd,
            )

    def _normalize_query_entry(self, entry: Any, *, tool_name: str) -> dict[str, Any]:
        if isinstance(entry, str):
            return {"prompt": entry, "start_s": None, "end_s": None}
        if not isinstance(entry, dict):
            raise TypeError(f"{tool_name} expects each query to be a string or dict")

        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{tool_name} requires a non-empty 'prompt'")

        start_s = entry.get("start_s")
        end_s = entry.get("end_s")
        if (start_s is None) != (end_s is None):
            raise ValueError(f"{tool_name} requires both start_s and end_s when scoping media")

        return {
            "prompt": prompt,
            "start_s": start_s,
            "end_s": end_s,
        }

    def _run_sub_llm_query(
        self,
        *,
        registry: ToolRegistry,
        repl: AgentREPL | None,
        prompt: str,
        start_s: float | None = None,
        end_s: float | None = None,
        source: str = "llm_query",
        batched: bool = False,
        budget: BudgetTracker | None = None,
        client: LLMClient | None = None,
        trace_depth: int = 0,
    ) -> str:
        llm_client = client or self._sub_llm
        budget = budget or self._budget

        if start_s is None and end_s is None:
            with self._tracer.llm_call(
                model=_model_label(self.sub_model),
                prompt=prompt,
                batched=batched,
                depth=trace_depth,
            ) as llm_trace:
                response = llm_client.completion(prompt)
                if repl is not None:
                    repl.record_llm_query(prompt, response)
                llm_trace.record_response(response)
                self._record_client_usage(
                    client=llm_client,
                    budget=budget,
                    default_model=_model_label(self.sub_model),
                    trace=llm_trace,
                )
            return response

        if start_s is None or end_s is None:
            raise ValueError("Media-bearing llm_query calls require both start_s and end_s")

        video_toolkit = self._find_video_toolkit(registry)
        if video_toolkit is None:
            raise ValueError("Media-bearing llm_query calls require an active VideoToolkit")

        request = video_toolkit.prepare_media_request(
            start_s=float(start_s),
            end_s=float(end_s),
            media_kind="video",
        )
        media_model = _model_label(llm_client.vision_model)
        trace_cm = video_toolkit._media_trace_context(
            kind=request["kind"],
            model=media_model,
            prompt=prompt,
            start_s=request["start_s"],
            end_s=request["end_s"],
            source=source,
        )

        with trace_cm as media_trace:
            response = llm_client.media_completion(
                prompt=prompt,
                media=request["media"],
            )
            if repl is not None:
                repl.record_llm_query(prompt, response)
            media_trace.record_response(response)
            media_trace.record(media_kind=request["kind"], artifact_path=request["artifact_path"])
            self._record_client_usage(
                client=llm_client,
                budget=budget,
                default_model=media_model,
                trace=media_trace,
            )

        video_toolkit.record_inspection(
            start_s=request["start_s"],
            end_s=request["end_s"],
            prompt=prompt,
            response=response,
            artifact_path=request["artifact_path"],
            kind=request["kind"],
            source=source,
            model=media_model,
        )
        return response

    def ask(
        self,
        question: str,
        *,
        context: Any = None,
        video: str | None = None,
        subtitle: str | None = None,
        document: str | list[str] | None = None,
        image: str | list[str] | None = None,
    ) -> Answer:
        """Run the RLM loop and return a structured answer."""
        start_time = time.time()
        loop_result: LoopResult | None = None
        evidence: list[Evidence] = []
        answer: Answer | None = None
        run_error: Exception | None = None

        # Auto-register VideoToolkit if video= provided and none registered
        if video and not self._has_video_toolkit():
            from .tools.video import VideoToolkit
            vt = VideoToolkit()
            vt._llm_client = self._sub_llm
            vt._audio_llm_client = self._audio_llm
            vt._tracer = self._tracer
            vt._budget = self._budget
            vt._prompt_config = self._prompts
            self._registry.register_toolkit(vt)

        # Auto-register DocumentToolkit if document= provided and none registered
        if document and not self._has_document_toolkit():
            from .tools.document import DocumentToolkit
            dt = DocumentToolkit()
            dt._llm_client = self._sub_llm
            dt._tracer = self._tracer
            dt._budget = self._budget
            dt._prompt_config = self._prompts
            self._registry.register_toolkit(dt)

        # Auto-register ImageToolkit if image= provided and none registered
        if image and not self._has_image_toolkit():
            from .tools.image import ImageToolkit
            it = ImageToolkit()
            it._llm_client = self._sub_llm
            it._tracer = self._tracer
            it._budget = self._budget
            it._prompt_config = self._prompts
            if self._captioner is not None:
                it._captioner = self._captioner
            self._registry.register_toolkit(it)

        # Build context dict for toolkits (modality classified later, inside the
        # completion span, so the sub_llm call shows up under sanjaya.completion)
        context_dict: dict[str, Any] = {
            "question": question,
            "context": context,
            "video": video,
            "subtitle": subtitle,
            "document": document,
            "image": image,
            "modality": "balanced",
        }

        # Setup toolkits
        for toolkit in self._registry.toolkits:
            toolkit.setup(context_dict)

        # Build a fresh registry for this run with the active toolkits
        run_registry = self._build_runtime_registry()

        # Create REPL
        repl = AgentREPL(
            registry=run_registry,
            context=context,
        )

        # Set up OS access from video toolkit if available
        for toolkit in run_registry.toolkits:
            if hasattr(toolkit, "get_os_access"):
                os_access = toolkit.get_os_access()
                if os_access is not None:
                    repl.set_os_access(os_access)

        # Create builtin tool closures
        def _llm_query(prompt: str, start_s: float | None = None, end_s: float | None = None) -> str:
            return self._run_sub_llm_query(
                registry=run_registry,
                repl=repl,
                prompt=prompt,
                start_s=start_s,
                end_s=end_s,
                source="llm_query",
                budget=self._budget,
                trace_depth=0,
            )

        def _llm_query_batched(queries: list[Any]) -> list[str]:
            from concurrent.futures import ThreadPoolExecutor

            normalized = [self._normalize_query_entry(q, tool_name="llm_query_batched") for q in queries]
            max_workers = max(1, min(4, len(normalized)))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._run_sub_llm_query,
                        registry=run_registry,
                        repl=repl,
                        prompt=entry["prompt"],
                        start_s=entry["start_s"],
                        end_s=entry["end_s"],
                        source="llm_query_batched",
                        batched=True,
                        budget=self._budget,
                        client=self._make_sub_llm_client(name="sub_llm_batched"),
                        trace_depth=0,
                    )
                    for entry in normalized
                ]
                return [future.result() for future in futures]

        def _get_state() -> dict[str, Any]:
            state: dict[str, Any] = {
                "tools": [t.name for t in run_registry.all_tools()],
                "iteration": "in_progress",
                "budget": self._budget.summary(),
            }
            for toolkit in run_registry.toolkits:
                toolkit_state = toolkit.get_state()
                if toolkit_state:
                    state.update(toolkit_state)
            return state

        # Register builtins
        run_registry.register(make_context_tool(lambda: repl.context))
        run_registry.register(make_llm_query_tool(_llm_query))
        run_registry.register(make_llm_query_batched_tool(_llm_query_batched))
        run_registry.register(make_done_tool(repl.mark_done))
        run_registry.register(make_get_state_tool(_get_state))

        # Register rlm_query builtins (only when recursion is enabled)
        if self._max_depth > 1:
            def _rlm_query(prompt: str, start_s: float | None = None, end_s: float | None = None) -> str:
                if (start_s is None) != (end_s is None):
                    raise ValueError("Slice-scoped rlm_query calls require both start_s and end_s")
                return self._subcall(
                    prompt,
                    depth=1,
                    parent_run_registry=run_registry,
                    parent_context=context,
                    active_span=(float(start_s), float(end_s)) if start_s is not None and end_s is not None else None,
                )

            def _rlm_query_batched(queries: list[Any]) -> list[str]:
                from concurrent.futures import ThreadPoolExecutor
                normalized = [self._normalize_query_entry(q, tool_name="rlm_query_batched") for q in queries]
                with ThreadPoolExecutor(max_workers=max(1, min(4, len(normalized)))) as pool:
                    futures = [
                        pool.submit(
                            self._subcall,
                            entry["prompt"],
                            depth=1,
                            parent_run_registry=run_registry,
                            parent_context=context,
                            active_span=(
                                (float(entry["start_s"]), float(entry["end_s"]))
                                if entry["start_s"] is not None and entry["end_s"] is not None
                                else None
                            ),
                        )
                        for entry in normalized
                    ]
                    return [f.result() for f in futures]

            run_registry.register(make_rlm_query_tool(_rlm_query))
            run_registry.register(make_rlm_query_batched_tool(_rlm_query_batched))

        # Build system prompt
        toolkit_sections = [
            tk.prompt_section()
            for tk in run_registry.toolkits
            if tk.prompt_section()
        ]

        context_metadata: dict[str, Any] = {}
        if context is not None:
            context_metadata["context_type"] = type(context).__name__
            if hasattr(context, "__len__"):
                context_metadata["context_length"] = len(context)
        if video:
            context_metadata["video"] = video

        system_prompt = build_system_prompt(
            registry=run_registry,
            context_metadata=context_metadata,
            toolkit_sections=toolkit_sections,
            max_depth=self._max_depth,
        )

        # Run the loop
        config = LoopConfig(
            max_iterations=self.max_iterations,
            max_budget_usd=self.max_budget_usd,
            max_timeout_s=self.max_timeout_s,
            compaction_threshold=self.compaction_threshold,
            critic_threshold=self.critic_threshold,
        )

        model_name = _model_label(self.model)
        try:
            with self._tracer.completion(
                question=question,
                question_preview=question[:200],
                model=model_name,
                orchestrator_model=model_name,
                recursive_model=_model_label(self.sub_model),
                audio_model=_model_label(self.audio_model) if self.audio_model else None,
                video_path=video,
                max_iterations=self.max_iterations,
            ) as comp_trace:
                try:
                    # Classify question modality inside the span so the sub_llm call
                    # is nested under sanjaya.completion in traces.
                    if video:
                        from .core.schema import classify_question_modality
                        modality = classify_question_modality(question, self._sub_llm)
                        context_dict["modality"] = modality
                        # Update toolkit modality and rebuild the system prompt
                        # so the correct strategy prompt is used.
                        for toolkit in run_registry.toolkits:
                            if hasattr(toolkit, "_modality"):
                                toolkit._modality = modality
                        toolkit_sections = [
                            tk.prompt_section()
                            for tk in run_registry.toolkits
                            if tk.prompt_section()
                        ]
                        system_prompt = build_system_prompt(
                            registry=run_registry,
                            context_metadata=context_metadata,
                            toolkit_sections=toolkit_sections,
                            max_depth=self._max_depth,
                        )

                    # Generate or use provided answer schema
                    from .core.schema import generate_answer_schema, schema_to_prompt_section

                    if self._prompts.answer_schema is not None:
                        answer_schema = self._prompts.answer_schema
                    else:
                        with self._tracer._span("sanjaya.schema_generation", question_chars=len(question)):
                            answer_schema = generate_answer_schema(
                                question=question,
                                llm_client=self._sub_llm,
                            )
                    self._current_schema = answer_schema
                    system_prompt = system_prompt + "\n\n" + schema_to_prompt_section(answer_schema)

                    loop_result = run_loop(
                        orchestrator=self._orchestrator,
                        repl=repl,
                        system_prompt=system_prompt,
                        question=question,
                        config=config,
                        budget=self._budget,
                        tracer=self._tracer,
                        critic=self._critic,
                        answer_schema=answer_schema,
                        critic_prompt=self._prompts.critic,
                        trace_context={"depth": 0},
                    )

                    # Collect evidence from toolkits
                    for toolkit in run_registry.toolkits:
                        evidence.extend(toolkit.build_evidence())

                    # Build Answer
                    raw = loop_result.raw_answer
                    if isinstance(raw, dict):
                        text = raw.get("summary") or raw.get("answer") or str(raw)
                        data = raw
                    else:
                        text = str(raw)
                        data = None

                    answer = Answer(
                        question=question,
                        text=text,
                        data=data,
                        evidence=evidence,
                        iterations=loop_result.iterations_used,
                        cost_usd=self._budget.total_cost_usd,
                        input_tokens=self._budget.total_input_tokens,
                        output_tokens=self._budget.total_output_tokens,
                        wall_time_s=round(time.time() - start_time, 2),
                    )

                    # Record final answer on the completion span so run_end SSE
                    # event includes it (the UI reads final_answer from run_end).
                    is_forced = loop_result.iterations_used >= config.max_iterations
                    comp_trace.record_final_answer(text, forced=is_forced)
                    comp_trace.record(answer_preview=text[:200])

                    # Record tokens and cost on the top-level span.
                    # Use the budget's accumulated cost (which prices each call
                    # with its actual model) instead of re-pricing all tokens
                    # under the orchestrator model — that caused cost divergence
                    # between the UI and Logfire.
                    comp_trace.record(
                        sanjaya_cost_usd=self._budget.total_cost_usd,
                        sanjaya_input_tokens=self._budget.total_input_tokens,
                        sanjaya_output_tokens=self._budget.total_output_tokens,
                    )
                except Exception as exc:
                    run_error = exc
                    comp_trace.record_error(exc)
                    raise
                finally:
                    for toolkit in run_registry.toolkits:
                        try:
                            toolkit.teardown()
                        except Exception as exc:
                            if run_error is None:
                                run_error = exc
                            comp_trace.record_error(exc)
        finally:
            try:
                self._persist_trace(
                    question=question,
                    loop_result=loop_result,
                    evidence=evidence,
                    error=run_error,
                    wall_time_s=round(time.time() - start_time, 2),
                )
            except Exception:
                if run_error is None:
                    raise

        if answer is None:
            raise run_error or RuntimeError("Agent run ended without an answer")

        self._last_answer = answer
        return answer

    # Names of builtin tools that are re-created per child (not inherited)
    _BUILTIN_NAMES = frozenset({
        "llm_query", "llm_query_batched",
        "rlm_query", "rlm_query_batched",
        "done", "get_context", "get_state",
    })

    def _subcall(
        self,
        prompt: str,
        *,
        depth: int,
        parent_run_registry: ToolRegistry,
        parent_context: Any = None,
        active_span: tuple[float, float] | None = None,
    ) -> str:
        """Run a recursive RLM sub-call with its own REPL and loop.

        At leaf depth (depth >= max_depth), falls back to a plain LLM
        completion with no REPL.
        """
        from rich.console import Console
        _console = Console()

        child_model_name = _model_label(self.sub_model)
        with self._tracer.subcall(
            depth=depth,
            prompt=prompt,
            child_model=child_model_name,
            start_s=active_span[0] if active_span is not None else None,
            end_s=active_span[1] if active_span is not None else None,
        ) as subcall_trace:
            try:
                if active_span is not None:
                    video_toolkit = self._find_video_toolkit(parent_run_registry)
                    if video_toolkit is None:
                        raise ValueError("Slice-scoped rlm_query calls require an active VideoToolkit")
                    video_toolkit._validate_span(
                        start_s=active_span[0],
                        end_s=active_span[1],
                        allow_zero=True,
                    )

                if depth >= self._max_depth:
                    _console.print(f"[dim]rlm_query d{depth}: leaf node, falling back to llm_query[/]")
                    subcall_trace.record(mode="leaf", leaf_node=True)
                    response = self._run_sub_llm_query(
                        registry=parent_run_registry,
                        repl=None,
                        prompt=prompt,
                        start_s=active_span[0] if active_span is not None else None,
                        end_s=active_span[1] if active_span is not None else None,
                        source="rlm_query_leaf",
                        budget=self._budget,
                        trace_depth=depth,
                    )
                    subcall_trace.record_response(response)
                    subcall_trace.record(status="complete", iterations_used=0)
                    return response

                _console.print(f"[bold magenta]>>> rlm_query d{depth}: spawning child loop[/]")
                subcall_trace.record(mode="recursive", leaf_node=False)

                child_toolkits = self._spawn_child_toolkits(
                    parent_registry=parent_run_registry,
                    active_span=active_span,
                    depth=depth,
                )
                inherited_tools = [
                    tool
                    for tool in parent_run_registry.all_tools()
                    if tool.name not in self._BUILTIN_NAMES
                ]
                child_registry = self._build_runtime_registry(
                    toolkits=child_toolkits,
                    standalone_tools=inherited_tools,
                )

                child_context = parent_context
                if isinstance(parent_context, dict):
                    child_context = dict(parent_context)
                elif active_span is not None:
                    child_context = {"parent_context": parent_context}
                if active_span is not None and isinstance(child_context, dict):
                    child_context["active_video_span"] = {
                        "start_s": active_span[0],
                        "end_s": active_span[1],
                    }

                repl = AgentREPL(registry=child_registry, context=child_context)

                for toolkit in child_registry.toolkits:
                    if hasattr(toolkit, "get_os_access"):
                        os_access = toolkit.get_os_access()
                        if os_access is not None:
                            repl.set_os_access(os_access)

                remaining_budget = None
                if self._budget.max_budget_usd is not None:
                    remaining_budget = max(0.0, self._budget.max_budget_usd - self._budget.total_cost_usd)

                remaining_timeout = None
                if self._budget.max_timeout_s is not None:
                    remaining_timeout = max(0.0, self._budget.max_timeout_s - self._budget.elapsed_s)

                child_budget = BudgetTracker(
                    max_budget_usd=remaining_budget,
                    max_timeout_s=remaining_timeout,
                )

                def _child_llm_query(p: str, start_s: float | None = None, end_s: float | None = None) -> str:
                    return self._run_sub_llm_query(
                        registry=child_registry,
                        repl=repl,
                        prompt=p,
                        start_s=start_s,
                        end_s=end_s,
                        source="llm_query",
                        budget=child_budget,
                        trace_depth=depth,
                    )

                def _child_llm_query_batched(queries: list[Any]) -> list[str]:
                    from concurrent.futures import ThreadPoolExecutor

                    normalized = [self._normalize_query_entry(q, tool_name="llm_query_batched") for q in queries]
                    with ThreadPoolExecutor(max_workers=max(1, min(4, len(normalized)))) as pool:
                        futures = [
                            pool.submit(
                                self._run_sub_llm_query,
                                registry=child_registry,
                                repl=repl,
                                prompt=entry["prompt"],
                                start_s=entry["start_s"],
                                end_s=entry["end_s"],
                                source="llm_query_batched",
                                batched=True,
                                budget=child_budget,
                                client=self._make_sub_llm_client(name=f"child_sub_llm_d{depth}_batched"),
                                trace_depth=depth,
                            )
                            for entry in normalized
                        ]
                        return [future.result() for future in futures]

                def _child_rlm_query(p: str, start_s: float | None = None, end_s: float | None = None) -> str:
                    if (start_s is None) != (end_s is None):
                        raise ValueError("Slice-scoped rlm_query calls require both start_s and end_s")
                    return self._subcall(
                        p,
                        depth=depth + 1,
                        parent_run_registry=child_registry,
                        parent_context=child_context,
                        active_span=(float(start_s), float(end_s)) if start_s is not None and end_s is not None else None,
                    )

                def _child_rlm_query_batched(queries: list[Any]) -> list[str]:
                    from concurrent.futures import ThreadPoolExecutor
                    normalized = [self._normalize_query_entry(q, tool_name="rlm_query_batched") for q in queries]
                    with ThreadPoolExecutor(max_workers=max(1, min(4, len(normalized)))) as pool:
                        futures = [
                            pool.submit(
                                self._subcall,
                                entry["prompt"],
                                depth=depth + 1,
                                parent_run_registry=child_registry,
                                parent_context=child_context,
                                active_span=(
                                    (float(entry["start_s"]), float(entry["end_s"]))
                                    if entry["start_s"] is not None and entry["end_s"] is not None
                                    else None
                                ),
                            )
                            for entry in normalized
                        ]
                        return [f.result() for f in futures]

                def _child_get_state() -> dict[str, Any]:
                    state: dict[str, Any] = {
                        "depth": depth,
                        "max_depth": self._max_depth,
                        "tools": [t.name for t in child_registry.all_tools()],
                        "budget": child_budget.summary(),
                    }
                    for toolkit in child_registry.toolkits:
                        toolkit_state = toolkit.get_state()
                        if toolkit_state:
                            state.update(toolkit_state)
                    return state

                child_registry.register(make_context_tool(lambda: repl.context))
                child_registry.register(make_llm_query_tool(_child_llm_query))
                child_registry.register(make_llm_query_batched_tool(_child_llm_query_batched))
                child_registry.register(make_done_tool(repl.mark_done))
                child_registry.register(make_get_state_tool(_child_get_state))
                child_registry.register(make_rlm_query_tool(_child_rlm_query))
                child_registry.register(make_rlm_query_batched_tool(_child_rlm_query_batched))

                context_metadata: dict[str, Any] = {}
                if active_span is not None:
                    context_metadata["active_video_span"] = f"{active_span[0]:.1f}s-{active_span[1]:.1f}s"

                toolkit_sections = [
                    tk.prompt_section()
                    for tk in child_registry.toolkits
                    if tk.prompt_section()
                ]

                system_prompt = build_system_prompt(
                    registry=child_registry,
                    context_metadata=context_metadata,
                    toolkit_sections=toolkit_sections,
                    max_depth=self._max_depth,
                )

                child_orchestrator = LLMClient(
                    model=self.sub_model,
                    vision_model=self.vision_model or self.sub_model,
                    fallback_model=None,
                    name=f"child_rlm_d{depth}",
                )

                child_max_iters = min(self.max_iterations, 10)

                child_config = LoopConfig(
                    max_iterations=child_max_iters,
                    max_budget_usd=remaining_budget,
                    max_timeout_s=remaining_timeout,
                    compaction_threshold=self.compaction_threshold,
                )

                result = run_loop(
                    orchestrator=child_orchestrator,
                    repl=repl,
                    system_prompt=system_prompt,
                    question=prompt,
                    config=child_config,
                    budget=child_budget,
                    tracer=self._tracer,
                    critic=None,
                    answer_schema=None,
                    trace_context={"depth": depth},
                )

                self._budget.record(
                    input_tokens=child_budget.total_input_tokens,
                    output_tokens=child_budget.total_output_tokens,
                    cost_usd=child_budget.total_cost_usd,
                    model=f"child_rlm_d{depth}",
                )

                _console.print(f"[bold magenta]<<< rlm_query d{depth}: child done ({result.iterations_used} iters)[/]")

                raw = result.raw_answer
                answer_text = str(raw.get("summary") or raw.get("answer") or raw) if isinstance(raw, dict) else str(raw)
                subcall_trace.record_response(answer_text)
                subcall_trace.record(
                    status="complete",
                    iterations_used=result.iterations_used,
                    child_cost_usd=child_budget.total_cost_usd,
                    child_input_tokens=child_budget.total_input_tokens,
                    child_output_tokens=child_budget.total_output_tokens,
                )
                return answer_text
            except Exception as exc:
                subcall_trace.record(status="error")
                subcall_trace.record_error(exc)
                raise

    @property
    def last_answer(self) -> Answer | None:
        """Most recent answer, for notebook inspection."""
        return self._last_answer

    @property
    def cost_so_far(self) -> float:
        """Cumulative USD spent across all ask() calls."""
        return self._budget.total_cost_usd

    def reset(self) -> None:
        """Clear all state (budget, history, workspace)."""
        self._budget = BudgetTracker(
            max_budget_usd=self.max_budget_usd,
            max_timeout_s=self.max_timeout_s,
        )
        self._last_answer = None
        self._registry = ToolRegistry()
        self._tracer = Tracer(enabled=self._tracer._enabled_requested, track_events=True)

    def _persist_trace(
        self,
        question: str,
        loop_result: LoopResult | None,
        evidence: list[Evidence],
        *,
        error: Exception | None = None,
        wall_time_s: float | None = None,
    ) -> None:
        """Write trace.json alongside clips/frames in the workspace."""
        import json

        workspace = None
        for toolkit in self._registry.toolkits:
            if hasattr(toolkit, "_workspace") and toolkit._workspace is not None:
                workspace = toolkit._workspace
                break

        if workspace is None:
            return

        model_name = _model_label(self.model)
        sub_model_name = _model_label(self.sub_model)
        vision_label = _model_label(self.vision_model) if self.vision_model else sub_model_name
        audio_label = _model_label(self.audio_model) if self.audio_model else None

        raw = loop_result.raw_answer if loop_result is not None else None
        trace = {
            "run_id": workspace.run_id,
            "question": question,
            "model": model_name,
            "sub_model": sub_model_name,
            "vision_model": vision_label,
            "audio_model": audio_label,
            "status": "error" if error is not None else "complete",
            "error": str(error) if error is not None else None,
            "error_type": type(error).__name__ if error is not None else None,
            "answer": str(raw),
            "answer_data": raw if isinstance(raw, dict) else None,
            "answer_schema": getattr(self, "_current_schema", None),
            "iterations": loop_result.iterations_used if loop_result is not None else None,
            "wall_time_s": round(loop_result.wall_time_s, 2) if loop_result is not None else wall_time_s,
            "cost": self._budget.summary(),
            "evidence_count": len(evidence),
            "events": self._tracer.dump_events(),
            "messages": loop_result.messages if loop_result is not None else [],
        }

        trace_path = workspace.run_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")

        workspace.record_trace_events(self._tracer.dump_events())

    def _has_video_toolkit(self) -> bool:
        """Check if a VideoToolkit is already registered."""
        try:
            from .tools.video import VideoToolkit
            return any(isinstance(tk, VideoToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False

    def _has_document_toolkit(self) -> bool:
        """Check if a DocumentToolkit is already registered."""
        try:
            from .tools.document import DocumentToolkit
            return any(isinstance(tk, DocumentToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False

    def _has_image_toolkit(self) -> bool:
        """Check if an ImageToolkit is already registered."""
        try:
            from .tools.image import ImageToolkit
            return any(isinstance(tk, ImageToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False
