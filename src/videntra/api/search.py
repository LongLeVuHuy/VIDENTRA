"""Search API routes and request/response models."""
from dataclasses import asdict
import time
from typing import Any, Literal
import uuid

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from videntra.query.rules import RuleBasedQueryParser
from videntra.query.schemas import QueryGraph
from videntra.retrieval.branches import VisualBranch, OCRBranch, ASRBranch, BranchCandidate
from videntra.retrieval.fusion import MultimodalMomentFusion
from videntra.ranking.ranker import RelationAwareRanker
from videntra.localization.refine import refine_moment_boundaries


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Max moments to return")
    mode: Literal["fast", "accurate"] = Field(default="fast", description="Search mode")
    ranking_mode: Literal["moment", "diverse_video"] = Field(default="moment")
    filters: dict[str, Any] = Field(default_factory=dict, description="Optional filters")
    client_request_id: str | None = Field(default=None, description="Client request ID for cancellation")
    verify_relations: bool = False
    rerank_with_relations: bool = False


class SearchResponse(BaseModel):
    request_id: str
    snapshot_id: str
    status: str
    parser_status: str
    branch_statuses: dict[str, str]
    query_graph: dict[str, Any] | None = None
    results: list[dict[str, Any]]
    warnings: list[str]
    timings: dict[str, Any]
    next_cursor: str | None = None
    truncated: bool = False
    synthetic: bool | None = None


router = APIRouter(prefix="/v1", tags=["search"])


@router.get("/query/parse")
def parse_query_endpoint(query: str) -> dict[str, Any]:
    """Parse a natural language query into validated QueryGraph JSON."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    parser = RuleBasedQueryParser()
    graph = parser.parse(query)
    return graph.to_dict()


@router.post("/search", response_model=SearchResponse)
def execute_search(req: SearchRequest, request: Request) -> SearchResponse:
    t0 = time.perf_counter()
    req_id = str(uuid.uuid4())
    pipeline = getattr(request.app.state, "pipeline", None)
    engine = getattr(request.app.state, "engine", None)
    text_store = getattr(request.app.state, "text_store", None)
    evidence_store = getattr(request.app.state, "evidence_store", None)
    snapshot_id = getattr(request.app.state, "snapshot_id", "corpus_current")

    if pipeline is None and (engine is None or len(engine.index_store) == 0):
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return SearchResponse(
            request_id=req_id,
            snapshot_id=snapshot_id,
            status="empty",
            parser_status="fallback",
            branch_statuses={"visual": "missing_index", "ocr": "not_requested", "asr": "not_requested"},
            query_graph=None,
            results=[],
            warnings=["Index or pipeline is not loaded or contains no vectors."],
            timings={"elapsed_ms": elapsed_ms, "measured": True},
            truncated=False,
            synthetic=False,
        )

    branch_statuses = {"visual": "complete", "ocr": "not_run", "asr": "not_run"}
    query_graph_dict = None
    warnings = []

    if pipeline is not None:
        warnings.append("Snapshot provenance and frame timestamp mappings are not verified by this endpoint.")
        # Single Source of Truth pipeline execution
        parser = RuleBasedQueryParser()
        query_graph = parser.parse(req.query)
        query_graph_dict = query_graph.to_dict()
        parser_status = query_graph.parser.status

        search_result = pipeline.search_with_metadata(
            snapshot_id=snapshot_id,
            verify_relations=req.verify_relations and req.mode == "accurate",
            rerank_with_relations=req.rerank_with_relations,
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            ranking_mode=req.ranking_mode,
            filters=req.filters,
        )
        moments = search_result["moments"]
        branch_statuses["visual"] = "complete"
        lexical_status = search_result["diagnostics"].get("lexical_status", "not_run")
        branch_statuses["ocr"] = lexical_status
        branch_statuses["asr"] = lexical_status
        if req.mode == "accurate" and not req.verify_relations:
            warnings.append("Relation verifier was not requested.")
        if req.mode != "accurate" and req.verify_relations:
            warnings.append("Relation verifier requires accurate mode and was not run.")
        from videntra.retrieval.unified_pipeline import unified_result_dict
        results = [unified_result_dict(m, snapshot_id) for m in moments]
        for result, assessment in zip(results, search_result["relation_assessments"]):
            result["relation_assessment"] = assessment

    elif req.mode == "accurate":
        # Legacy fallback mode with engine and text_store
        parser = RuleBasedQueryParser()
        query_graph = parser.parse(req.query)
        query_graph_dict = query_graph.to_dict()
        parser_status = query_graph.parser.status

        # 2. Retrieve Visual candidates
        vis_branch = VisualBranch(engine)
        vis_cands, vis_status, _ = vis_branch.retrieve(req.query, top_k=req.top_k * 2)
        branch_statuses["visual"] = vis_status

        # 3. Retrieve Text candidates if TextStore is active
        ocr_candidates = []
        asr_candidates = []
        if text_store is not None:
            ocr_b = OCRBranch(text_store)
            ocr_candidates, ocr_status, _ = ocr_b.retrieve(req.query, top_k=req.top_k)
            branch_statuses["ocr"] = ocr_status

            asr_b = ASRBranch(text_store)
            asr_candidates, asr_status, _ = asr_b.retrieve(req.query, top_k=req.top_k)
            branch_statuses["asr"] = asr_status

        # 4. Multimodal Fusion (RRF)
        fusion = MultimodalMomentFusion()
        fused_moments = fusion.fuse(
            visual_candidates=vis_cands,
            ocr_candidates=ocr_candidates,
            asr_candidates=asr_candidates,
            top_k=req.top_k * 2,
        )

        # 5. Collect evidence items for verification
        evidence_by_video: dict[str, list[dict[str, Any]]] = {}
        for c in list(vis_cands) + list(ocr_candidates) + list(asr_candidates):
            if c.evidence_item:
                evidence_by_video.setdefault(c.video_id, []).append(c.evidence_item)

        # 6. Relation-Aware Reranking with GraphMatcher
        ranker = RelationAwareRanker()
        moments = ranker.rank(
            moments=fused_moments,
            query_graph=query_graph,
            evidence_by_video=evidence_by_video,
        )[:req.top_k]
        results = [m.to_result_dict() for m in moments]

    else:
        # Fast mode: visual vector search only
        parser_status = "not_run"
        moments = engine.search(
            query=req.query,
            top_k_moments=req.top_k,
        )
        results = [m.to_result_dict() for m in moments]


    # Diverse video filtering if requested
    if pipeline is None and req.ranking_mode == "diverse_video":
        seen_vids = set()
        diverse_results = []
        for r in results:
            if r["video_id"] not in seen_vids:
                seen_vids.add(r["video_id"])
                diverse_results.append(r)
        results = diverse_results[:req.top_k]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    status = "complete" if results else "no_supported_match"

    return SearchResponse(
        request_id=req_id,
        snapshot_id=snapshot_id,
        status=status,
        parser_status=parser_status,
        branch_statuses=branch_statuses,
        query_graph=query_graph_dict,
        results=results,
        warnings=warnings,
        timings={"elapsed_ms": elapsed_ms, "measured": True, **(search_result["timings"] if pipeline is not None else {})},
        truncated=len(moments) >= req.top_k,
        synthetic=None if pipeline is not None else False,
    )
