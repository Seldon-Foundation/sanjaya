"""FastAPI application for Sanjaya HUD backend."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load the repo-level .env so model provider credentials are available even
# when the API is started from a shell that has not exported them.
load_dotenv(PROJECT_ROOT / ".env", override=True)

from sanjaya_api.routes.benchmark_jobs import router as benchmark_jobs_router
from sanjaya_api.routes.health import router as health_router
from sanjaya_api.routes.mmou_jobs import router as mmou_jobs_router
from sanjaya_api.routes.runs import router as runs_router

app = FastAPI(
    title="Sanjaya API",
    description="FastAPI bridge for VideoRLM orchestration monitoring",
    version="0.1.0",
)

# CORS — allow the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5100",
        "http://127.0.0.1:5100",
        "http://localhost:5101",
        "http://127.0.0.1:5101",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(runs_router)
app.include_router(benchmark_jobs_router)
app.include_router(mmou_jobs_router)
