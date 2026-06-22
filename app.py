"""
app.py

FastAPI service exposing the unified FinFarm LangGraph pipeline.

Endpoints
---------
GET  /            → service banner
GET  /health      → liveness + whether models are warm
POST /query       → run the full pipeline for one query

Models are loaded once at startup (registry.warmup()) so the first real
request isn't penalised with multi-GB cold loads. Designed to run as the
Docker container on a HuggingFace Space (port 7860).
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared import registry
from shared.utils import get_logger
from pipeline.graph import run as run_pipeline

logger = get_logger(__name__)

# Set True once warmup() finishes (used by /health).
_MODELS_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm all models before the server starts accepting traffic."""
    global _MODELS_READY
    try:
        registry.warmup()
        _MODELS_READY = True
        logger.info("Startup warmup complete — service ready.")
    except Exception as exc:  # pragma: no cover - surfaced via /health
        # Don't crash the server; /health will report not-ready and the first
        # request will retry the (lazy) loads.
        logger.error("Warmup failed at startup: %s", exc)
    yield


app = FastAPI(
    title="FinFarm AI",
    description="Multilingual Agro / Banking / General query assistant.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query in any supported language.")
    domain: Literal["agro", "banking", "general"] = Field(
        ..., description="Branch selected by the user."
    )
    state_name: Optional[str] = Field(
        None, description="Optional KCC state filter (agro branch only), e.g. TAMILNADU."
    )


class QueryResponse(BaseModel):
    final_answer: str
    detected_lang: str
    domain: str
    intent: Optional[str] = None
    labels: Optional[list[str]] = None
    answers_en: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "FinFarm AI", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": _MODELS_READY}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = run_pipeline(req.query, domain=req.domain, state_name=req.state_name)
    except Exception as exc:
        logger.exception("Pipeline failed for query=%r", req.query)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return QueryResponse(
        final_answer=result.get("final_answer", ""),
        detected_lang=result.get("detected_lang", ""),
        domain=req.domain,
        intent=result.get("intent"),
        labels=result.get("labels"),
        answers_en=result.get("answers_en"),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
