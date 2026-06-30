set shell := ["bash", "-cu"]

# Start API (port 8000) + UI (port 5100) together via Overmind.
dev:
	overmind start

# Start only the FastAPI backend (port 8000).
api:
	uv run --project api uvicorn sanjaya_api.main:app --port 8000 --reload

# Start only the Next.js UI (port 5100).
ui:
	cd ui && bun dev --port 5100

# Run MMOU from the CLI. Pass extra args after the recipe name.
mmou *args:
	uv run sanjaya-mmou {{args}}

# Inspect latest persisted trace (or pass manifest/run_id).
video-trace manifest="" run_id="":
	if [[ -n "{{manifest}}" ]]; then uv run python scripts/inspect_video_trace.py --manifest "{{manifest}}"; elif [[ -n "{{run_id}}" ]]; then uv run python scripts/inspect_video_trace.py --run-id "{{run_id}}"; else uv run python scripts/inspect_video_trace.py; fi
