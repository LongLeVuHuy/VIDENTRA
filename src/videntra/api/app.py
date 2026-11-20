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
from videntra.config.device import get_device_info


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
        engine_obj = getattr(app.state, "engine", None)
        pipeline_obj = getattr(app.state, "pipeline", None)
        has_engine = engine_obj is not None
        has_pipeline = pipeline_obj is not None
        has_resolver = getattr(app.state, "resolver", None) is not None
        if has_pipeline:
            aug_emb = getattr(pipeline_obj, "augmented_embeddings", None)
            vec_count = int(len(aug_emb)) if aug_emb is not None else 0
            kf_map = getattr(pipeline_obj, "keyframe_map", None) or {}
            vid_durs = getattr(pipeline_obj, "video_durations", None) or {}
            vid_count = len(vid_durs) if vid_durs else len({k.get("video_id") for k in kf_map.values() if isinstance(k, dict) and k.get("video_id")})
            vp_ret = getattr(pipeline_obj, "videoprism_retriever", None)
            vp_status = getattr(vp_ret, "status", "disabled") if getattr(pipeline_obj, "enable_videoprism", False) else "disabled"
            ltr_loaded = bool(getattr(getattr(pipeline_obj, "ltr_ranker", None), "is_loaded", False))
            backend = "unified"
        else:
            vec_count = len(engine_obj.index_store) if has_engine else 0
            vid_count = vec_count
            vp_status = "unavailable"
            ltr_loaded = False
            backend = "visual" if has_engine else "none"
        active_device = (
            getattr(pipeline_obj, "device", None)
            or getattr(getattr(engine_obj, "encoder", None), "device", None)
        )
        return {
            "status": "ok",
            "service": "videntra",
            "version": "0.1.0",
            "snapshot_id": app.state.snapshot_id,
            "backend": backend,
            "device": active_device,
            "cuda_available": bool(get_device_info()["cuda_available"]),
            "engine_active": bool(has_engine or has_pipeline),
            "pipeline_active": has_pipeline,
            "indexed_samples": vec_count,
            "indexed_vectors": vec_count,
            "indexed_videos": vid_count,
            "videoprism_status": vp_status,
            "ltr_loaded": ltr_loaded,
            "resolver_active": has_resolver,
        }

    @app.get("/", response_class=HTMLResponse, tags=["ui"])
    def render_ui():
        return HTMLResponse(content=UI_HTML)

    return app
