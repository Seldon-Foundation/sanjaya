# Sanjaya

Sanjaya is a video-only RLM agent runner for MMOU-style video question answering. It launches runs from a small viewer or from a CLI, uses Gemini for agent and video/audio reasoning, and prepares an OpenAI speech-to-text transcript before each video reaches the agent.

The current product surface is intentionally small:

- MMOU job launcher in the web viewer.
- MMOU CLI runner for headless runs.
- Recursive language model agent with video inspection, delegated LLM calls, delegated child agents, zoom, audio analysis, and a spoken-language transcript.

## RLM Architecture

RLM means Recursive Language Model. Instead of asking one model call to solve a whole video question in a single prompt, Sanjaya runs a root agent in a Python REPL loop:

1. The root model receives the sanitized question, tool docs, and lightweight runtime state.
2. It writes one Python code block.
3. The sandbox executes the code and returns observations.
4. The model sees those observations on the next iteration.
5. It repeats until it calls `done(value)`.

The REPL is deliberately constrained. It gives the model normal Python data structures and the Sanjaya tools, but no file I/O, shell, imports beyond the allowed standard modules, or network access. That keeps the agent focused on the provided tools and makes traces easier to inspect.

### Root Agent

The root agent lives in `src/sanjaya/agent.py`. It is responsible for:

- Building the system prompt from core RLM instructions, registered tool docs, video-toolkit notes, transcript notes, and answer schema instructions.
- Preparing a transcript before the first model iteration when a video is present.
- Registering built-ins such as `llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched`, `done`, and `get_state`.
- Registering video tools from `VideoToolkit`.
- Running the iterative model/code/observe loop.
- Tracking token usage, budget, timeout, trace events, and final answers.

The root agent does not receive MMOU metadata such as evidence windows or benchmark framing. The MMOU adapter sanitizes the prompt before calling the agent.

### Video Tools

Video behavior is implemented in `src/sanjaya/tools/video/`.

The main tools exposed to the REPL are:

- `inspect_video(start_s, end_s, zoom_box=None)`: queues a short video slice or frame crop into the root model's next turn. It is promptless; it does not ask a separate model question.
- `analyze_audio(start_s, end_s)`: asks the audio model to summarize spoken/audio content for an explicit slice.
- `llm_query(prompt, start_s=None, end_s=None, zoom_box=None)`: makes a one-shot sub-LLM call. With no times it is text-only. With times it sends exactly that video slice.
- `llm_query_batched([...])`: runs independent `llm_query` calls concurrently.
- `rlm_query(prompt, start_s=None, end_s=None, zoom_box=None)`: spawns a child RLM agent with its own REPL and tools. With times it is scoped to that slice.
- `rlm_query_batched([...])`: runs multiple child agents concurrently.

`zoom_box` uses `(x1, y1, x2, y2)` coordinates on a 0-1000 grid from the top-left of the visible frame. The crop is enlarged to fill the frame for the receiving model. This is meant for tiny text, UI elements, player reactions, scoreboards, and other visually ambiguous details.

### Transcript Flow

Before a video run starts, `src/sanjaya/tools/video/transcription.py` extracts audio with `ffmpeg` and calls the OpenAI speech-to-text API. The current default is `whisper-1`, which provides segment timestamps. Empty transcripts are allowed, because some videos have no speech.

The transcript is injected into the REPL as a variable named `transcript` with this shape:

```python
{
    "text": "full spoken-language transcript",
    "segments": [
        {"start": 12.34, "end": 15.67, "text": "spoken words in this segment"}
    ],
    "metadata": {...},
}
```

The prompt tells the agent that this transcript is for spoken language only, not music or sound effects. The transcript text is available to the REPL, but it is not pasted directly into every model prompt.

### Delegation Model

Sanjaya uses three levels of model work:

- Root model: orchestrates the run, chooses evidence to inspect, verifies delegated claims, and returns the final answer.
- Sub model: handles direct `llm_query` requests, either text-only or with one explicit video slice.
- Child RLM agents: handle `rlm_query` tasks that need their own tool loop, such as investigating a slice over multiple steps.

Batched calls are important for throughput. `llm_query_batched` and `rlm_query_batched` run independent work concurrently inside one agent iteration.

### Tracing And Artifacts

Trace events are recorded by `src/sanjaya/tracing/`. They include model calls, tool calls, transcript preparation, media attachments, costs, errors, and final answers. The viewer streams these events live and can inspect persisted traces after a run.

MMOU run artifacts are written by the external `videobench` repository layout. Sanjaya also writes per-question artifacts when `--keep-artifacts` or the matching UI option is enabled.

## Code Tour

Core agent code:

- `src/sanjaya/agent.py`: top-level `Agent`, root loop setup, child-agent spawning, transcript injection, budget/timeout handling.
- `src/sanjaya/core/prompts.py`: concise RLM system prompts and recursive-agent instructions.
- `src/sanjaya/core/loop.py`: model/code/observe loop helpers.
- `src/sanjaya/core/repl.py`: restricted Python REPL state and execution.
- `src/sanjaya/core/schema.py`: structured answer schema generation.
- `src/sanjaya/tools/builtins.py`: built-in tool definitions for `llm_query`, `rlm_query`, batched variants, state, and finalization.
- `src/sanjaya/tools/registry.py`: tool and toolkit registration.

Video code:

- `src/sanjaya/tools/video/toolkit.py`: `inspect_video`, `analyze_audio`, clip/frame extraction, zoom, prompt section, and trace payloads.
- `src/sanjaya/tools/video/transcription.py`: OpenAI transcript generation, sidecar loading, retries, timestamp normalization.
- `src/sanjaya/tools/video/native_audio.py`: native audio model calls.
- `src/sanjaya/tools/video/native_video.py`: native Gemini video calls and media preparation.

Model and configuration code:

- `src/sanjaya/llm/client.py`: pydantic-ai wrapper for text, media, batch calls, usage, retries, and cost metadata.
- `src/sanjaya/model_defaults.py`: default Gemini model names.
- `src/sanjaya/settings.py`: `.env` loading and environment settings.

MMOU code:

- `src/sanjaya/benchmarks/mmou_adapter.py`: adapter from `videobench` MMOU samples to Sanjaya agent calls, including prompt sanitization.
- `src/sanjaya/benchmarks/mmou_cli.py`: `sanjaya-mmou` CLI entry point.
- `api/sanjaya_api/services/mmou_jobs.py`: backend job service, workers, resume/stop/evaluate, persisted job hydration.
- `api/sanjaya_api/routes/mmou_jobs.py`: FastAPI routes.
- `api/sanjaya_api/models.py`: API request/response models.

Viewer code:

- `ui/app/page.tsx`: viewer entry page.
- `ui/components/benchmark/mmou-launcher.tsx`: MMOU launch, selection, run list, controls, and results UI.
- `ui/components/hud/trace-timeline.tsx`: trace event display.
- `ui/lib/api.ts`: browser API client for backend routes.
- `ui/lib/types.ts`: shared viewer types.

Utilities:

- `scripts/inspect_video_trace.py`: inspect persisted trace manifests from the terminal.
- `justfile`: local development commands.
- `Procfile`: Overmind process definitions for backend and UI.

## Setup

Install Python dependencies:

```bash
uv sync
```

Install viewer dependencies:

```bash
cd ui
bun install
```

If Bun is not available, npm also works for the current viewer:

```bash
cd ui
npm install
```

Create a root `.env` file:

```bash
OPENAI_API_KEY=...

# For Google AI Studio style auth:
GOOGLE_API_KEY=...

# Or for Vertex/service-account auth:
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service_account.json
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=global
```

The MMOU integration imports the external benchmarks repository. By default it expects:

```bash
/Users/lsteno/Developer/GitHub/benchmarks
```

Override that path with:

```bash
export SANJAYA_BENCHMARKS_DIR=/path/to/benchmarks
```

System tools required for video runs:

- `ffmpeg`
- `ffprobe`

## Run The Viewer

Start backend and viewer together with Overmind:

```bash
just dev
```

Or run them in separate terminals:

```bash
just api
just ui
```

If you are using npm instead of Bun for the viewer:

```bash
cd ui
npm run dev -- --port 5100
```

Open:

```text
http://localhost:5100
```

The backend runs at:

```text
http://localhost:8000
```

The viewer supports:

- Loading the MMOU catalog.
- Selecting domains, limits, or specific question IDs.
- Setting worker count, attempts, iteration limit, recursion depth, budgets, and timeouts.
- Launching, stopping, resuming, and evaluating jobs.
- Streaming live trace events.
- Inspecting per-question results and traces.

## Run MMOU From The CLI

Smoke test one question:

```bash
uv run sanjaya-mmou --limit 1 --workers 1
```

Equivalent `just` shortcut:

```bash
just mmou --limit 1 --workers 1
```

Run a larger balanced MMOU job:

```bash
uv run sanjaya-mmou \
  --run mmou_$(date +%Y%m%d_%H%M%S) \
  --domains all \
  --selection-mode balanced_domains \
  --per-domain-limit 300 \
  --workers 6 \
  --max-attempts 1 \
  --max-iterations 8 \
  --max-depth 2 \
  --max-timeout-s 900 \
  --keep-artifacts
```

Run one domain:

```bash
uv run sanjaya-mmou \
  --domains "Video Games / Esports Tournaments" \
  --limit 10 \
  --workers 6
```

Use specific models:

```bash
uv run sanjaya-mmou \
  --root-model google-vertex:gemini-3.1-pro-preview \
  --sub-model google-vertex:gemini-3-flash-preview \
  --recursive-model google-vertex:gemini-3.1-pro-preview \
  --vision-model google-vertex:gemini-3.1-pro-preview \
  --audio-model google-vertex:gemini-3-flash-preview \
  --limit 5
```

Export and evaluate:

```bash
uv run sanjaya-mmou \
  --run mmou_eval \
  --limit 20 \
  --workers 4 \
  --evaluate
```

Show all CLI options:

```bash
uv run sanjaya-mmou --help
```

## API Routes

The backend is intentionally narrow:

- `GET /health`
- `GET /mmou-jobs/catalog`
- `POST /mmou-jobs`
- `GET /mmou-jobs`
- `GET /mmou-jobs/{job_id}`
- `POST /mmou-jobs/{job_id}/stop`
- `POST /mmou-jobs/{job_id}/resume`
- `POST /mmou-jobs/{job_id}/evaluate`
- `POST /mmou-jobs/{job_id}/questions/{question_id}/evaluate`
- `GET /mmou-jobs/{job_id}/questions/{question_id}/trace`
- `GET /mmou-jobs/{job_id}/events`

## Inspect Traces

Inspect the latest trace:

```bash
just video-trace
```

Inspect by run id:

```bash
just video-trace run_id=mmou_20260630_212637
```

Inspect by manifest path:

```bash
just video-trace manifest=/path/to/manifest.json
```

The viewer is usually easier for browsing live traces; the script is useful when diagnosing a completed or failed run from the terminal.

## Development Checks

Python lint:

```bash
uv run ruff check src api tests
```

Focused Python tests:

```bash
uv run pytest \
  tests/test_video_native_rlm.py \
  tests/test_mmou_rlm_adapter.py \
  tests/test_mmou_job_service.py \
  tests/test_mmou_job_routes.py \
  tests/test_llm_client_retries.py
```

Full Python test suite:

```bash
uv run pytest
```

Viewer lint and build:

```bash
cd ui
bun run lint
bun run build
```

With npm:

```bash
cd ui
npm run lint
npm run build
```

## Adding Another Benchmark

MMOU is the only launch path right now, but the code is structured so another video benchmark can be added without changing the agent architecture.

Add new benchmark support by following the MMOU pattern:

1. Add an adapter under `src/sanjaya/benchmarks/` that converts benchmark samples into sanitized `Agent.ask(..., video=...)` calls.
2. Keep benchmark-specific metadata out of the agent prompt unless the model genuinely needs it to answer the question.
3. Add a CLI entry point or extend the current CLI with a benchmark selector.
4. Add backend service/routes if the benchmark should be launched from the viewer.
5. Add a viewer component that calls the backend service.

Do not reintroduce the old generic demo benchmark runner. The preferred interface is a small adapter per benchmark plus a unified launch surface.

## Operational Notes

- `llm_query()` without `start_s` and `end_s` is text-only.
- Media-bearing `llm_query()` and `rlm_query()` calls require explicit `start_s` and `end_s`; they do not silently receive the whole video.
- Automatic transcription happens before the agent starts.
- Transcription failures are retried with exponential backoff up to one minute.
- Empty transcripts are valid and are not treated as failures.
- Workers parallelize MMOU questions; each question still performs its own transcript, video, model, and child-agent work.
- More workers can increase throughput but also increases provider concurrency pressure and local `ffmpeg` work.
