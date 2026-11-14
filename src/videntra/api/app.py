"""FastAPI main application for VIDENTRA Search and Playback Inspector."""
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .search import router as search_router
from .media import router as media_router
from .frames import router as frames_router
from .exports import router as exports_router
from .ui_html import UI_HTML


def create_app(
    engine: Any = None,
    resolver: Any = None,
    text_store: Any = None,
    evidence_store: Any = None,
    ingest_dir: Path | str | None = None,
    snapshot_id: str = "corpus_current",
    pipeline: Any = None,
) -> FastAPI:
    """Instantiate configured FastAPI application."""
    app = FastAPI(
        title="VIDENTRA API",
        version="0.1.0",
        description="Multi-Modal Video Moment Retrieval Engine API",
    )

    # Allow CORS for development and local testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared engine and data context to application state
    app.state.engine = engine
    app.state.resolver = resolver
    app.state.text_store = text_store
    app.state.evidence_store = evidence_store
    app.state.ingest_dir = Path(ingest_dir) if ingest_dir else None
    app.state.snapshot_id = snapshot_id
    app.state.pipeline = pipeline


    # Register routers
    app.include_router(search_router)
    app.include_router(media_router)
    app.include_router(frames_router)
    app.include_router(exports_router)

    @app.get("/health", tags=["system"])
    def health_check():
        has_engine = getattr(app.state, "engine", None) is not None
        has_resolver = getattr(app.state, "resolver", None) is not None
        indexed_count = len(app.state.engine.index_store) if has_engine else 0
        return {
            "status": "ok",
            "service": "videntra",
            "version": "0.1.0",
            "snapshot_id": app.state.snapshot_id,
            "engine_active": has_engine,
            "indexed_samples": indexed_count,
            "indexed_vectors": indexed_count,
            "resolver_active": has_resolver,
        }

    @app.get("/", response_class=HTMLResponse, tags=["ui"])
    def render_ui():
        return HTMLResponse(content=UI_HTML)

    return app
