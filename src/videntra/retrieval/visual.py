"""Visual retrieval engine combining SigLIP 2 encoder, stable ID mapping, and FAISS index."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import time
from typing import Any, Sequence
import uuid

import numpy as np
import pyarrow.parquet as pq

from videntra.encoders.siglip2 import SigLIP2Encoder, MockSigLIP2Encoder
from videntra.indexing.ids import IDMappingStore, SampleCoordinate
from videntra.indexing.faiss_store import VisualIndexStore
from videntra.timeline.ledger import exact_seek_frame, read_frame_record
from videntra.query.rules import decompose_query_clauses


def ensemble_clause_scores(
    full_score: float,
    clause_scores: Sequence[float],
    tau_clause: float = 0.22,
    weight_full: float = 0.70,
    weight_clause: float = 0.30,
    coverage_bonus_scale: float = 0.05,
) -> float:
    """Combine full query score and sub-clause scores with coverage bonus."""
    if not clause_scores:
        return float(full_score)

    max_clause = max(clause_scores) if len(clause_scores) > 0 else 0.0
    combined = weight_full * full_score + weight_clause * max_clause

    num_hits = sum(1 for s in clause_scores if s > tau_clause)
    if num_hits >= 2 and len(clause_scores) > 0:
        bonus = coverage_bonus_scale * (num_hits / len(clause_scores))
        combined += bonus

    return float(combined)


def temporal_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate temporal Intersection over Union between two intervals [start, end)."""
    inter = max(0.0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class RankedMoment:
    moment_id: str
    video_id: str
    moment_start_seconds: float
    moment_end_seconds: float
    start_original_frame_index: int
    end_original_frame_index: int
    representative_original_frame_index: int
    representative_timestamp_seconds: float
    ranking_score: float
    score_type: str = "uncalibrated_ranking"
    match_status: str = "supported"
    evidence_frame_indices: tuple[int, ...] = ()
    unresolved_predicates: tuple[str, ...] = ()
    structural_score: float = 0.0

    def to_result_dict(
        self,
        media_uri: str = "",
        thumbnail_uri: str | None = None,
        source_video_id: str | None = None,
        time_base: tuple[int, int] = (1, 1000),
        timeline_revision: str = "rev1",
    ) -> dict[str, Any]:
        """Convert to canonical Result contract representation."""
        t_sec = float(self.representative_timestamp_seconds)
        start_sec = float(self.moment_start_seconds)
        end_sec = max(float(self.moment_end_seconds), t_sec + 0.001)

        rep_us = int(round(t_sec * 1_000_000))

        return {
            "video_id": self.video_id,
            "source_video_id": source_video_id or self.video_id,
            "moment_start_seconds": round(start_sec, 4),
            "moment_end_seconds": round(end_sec, 4),
            "start_original_frame_index": self.start_original_frame_index,
            "end_original_frame_index": self.end_original_frame_index,
            "representative_keyframe_id": f"{self.video_id}:frame_{self.representative_original_frame_index}",
            "representative_original_frame_index": self.representative_original_frame_index,
            "representative_timestamp_seconds": round(t_sec, 4),
            "ranking_score": round(float(self.ranking_score), 4),
            "score_type": self.score_type,
            "match_status": self.match_status,
            "unresolved_predicates": list(self.unresolved_predicates),
            "evidence": [
                {
                    "evidence_id": f"ev_{self.video_id}_{idx}",
                    "kind": "frame",
                    "original_decoded_frame_index": idx,
                    "timestamp_us": rep_us,
                }
                for idx in self.evidence_frame_indices
            ],
            "media_uri": media_uri,
            "thumbnail_uri": thumbnail_uri,
            "timing_metadata": {
                "frame_index_base": 0,
                "frame_index_scope": "ingested_video_presentation_order",
                "end_frame_inclusive": True,
                "interval_convention": "[start,end)",
                "coordinate_system": "clip_relative",
                "time_base": {"num": time_base[0], "den": time_base[1]},
                "first_video_pts": None,
                "representative_pts": None,
                "source_offset_seconds": None,
                "source_start_seconds": None,
                "source_end_seconds": None,
                "untrimmed_source_frame_index": None,
                "proxy_to_clip_offset_seconds": None,
                "timeline_revision": timeline_revision,
                "boundary_method": "windowed_sampling",
                "boundary_uncertainty_seconds": 0.2,
            },
            "provenance": {
                "synthetic": True,
                "index_snapshot": "snapshot_m2_01",
                "model_manifest": "siglip2_base_patch16_224",
                "query_graph_version": "1.0",
                "media_sha256": None,
                "run_id": "m2_visual_run",
            },
        }


class VisualRetrievalEngine:
    """End-to-end visual retrieval system from raw query to ranked video moments."""

    def __init__(
        self,
        encoder: SigLIP2Encoder | MockSigLIP2Encoder | None = None,
        index_store: VisualIndexStore | None = None,
        id_mapping: IDMappingStore | None = None,
    ):
        self.encoder = encoder or MockSigLIP2Encoder()
        self.index_store = index_store or VisualIndexStore(dimension=768)
        self.id_mapping = id_mapping or IDMappingStore()

    def index_video(
        self,
        video_id: str,
        media_path: Path | str,
        samples_parquet_path: Path | str,
        timeline_parquet_path: Path | str,
    ) -> int:
        """Extract sample frames, compute SigLIP 2 embeddings, and add to index."""
        media_p = Path(media_path)
        samples_p = Path(samples_parquet_path)
        timeline_p = Path(timeline_parquet_path)

        samples_table = pq.read_table(str(samples_p))
        ordinals = samples_table.column("original_decoded_frame_index").to_pylist()
        sample_ids = samples_table.column("sampled_frame_id").to_pylist()
        ts_us_list = samples_table.column("timestamp_us").to_pylist()
        ts_sec_list = samples_table.column("timestamp_seconds").to_pylist()
        tl_rev = samples_table.column("timeline_revision")[0].as_py() if "timeline_revision" in samples_table.column_names else "rev_m1_1"

        if not ordinals:
            return 0

        # Retrieve exact decoded pixels for each sampled frame
        pixels_list = []
        valid_indices = []
        for i, ord_idx in enumerate(ordinals):
            try:
                frame = exact_seek_frame(media_p, timeline_p, target_ordinal=ord_idx)
                if frame.pixels is not None:
                    pixels_list.append(frame.pixels)
                    valid_indices.append(i)
            except Exception:
                pass

        if not pixels_list:
            return 0

        # Compute embeddings in batches
        embeddings = self.encoder.encode_images(pixels_list)

        assigned_ids = []
        for local_idx in valid_indices:
            int_id = self.id_mapping.register(
                video_id=video_id,
                timeline_revision=tl_rev,
                ordinal=ordinals[local_idx],
                sampled_frame_id=sample_ids[local_idx],
                timestamp_us=ts_us_list[local_idx],
                timestamp_seconds=ts_sec_list[local_idx],
            )
            assigned_ids.append(int_id)

        # Add to vector index
        self.index_store.add(embeddings, assigned_ids)
        return len(assigned_ids)

    def search(
        self,
        query: str,
        top_k_moments: int = 10,
        window_duration_seconds: float = 2.0,
        nms_iou_threshold: float = 0.5,
        search_depth: int = 250,
        enable_ensemble: bool = True,
        tau_clause: float = 0.22,
    ) -> list[RankedMoment]:
        """Search corpus by text query, proposal generation, scoring, and temporal NMS."""
        if len(self.index_store) == 0:
            return []

        search_k = min(search_depth, len(self.index_store))
        clauses = decompose_query_clauses(query) if enable_ensemble else []

        # 1. Multi-clause query batch embedding & scoring
        sub_clauses = [c for c in clauses if c.strip().lower() != query.strip().lower()]
        if enable_ensemble and sub_clauses:
            all_queries = [query] + sub_clauses
            all_vecs = self.encoder.encode_texts(all_queries)
            scores_batch, ids_batch = self.index_store.search(all_vecs, top_k=search_k)

            full_scores = scores_batch[0]
            full_ids = ids_batch[0]
            full_map = dict(zip(full_ids, full_scores))

            clause_maps = [
                dict(zip(ids_batch[i + 1], scores_batch[i + 1]))
                for i in range(len(sub_clauses))
            ]

            candidate_ids = set(full_ids)
            for c_ids in ids_batch[1:]:
                candidate_ids.update(c_ids)

            scored_candidates = []
            for int_id in candidate_ids:
                s_full = float(full_map.get(int_id, 0.0))
                c_scores = [float(cm.get(int_id, 0.0)) for cm in clause_maps]
                s_comb = ensemble_clause_scores(
                    full_score=s_full,
                    clause_scores=c_scores,
                    tau_clause=tau_clause,
                )
                scored_candidates.append((int_id, s_comb))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            q_ids = [cid for cid, _ in scored_candidates]
            q_scores = [sc for _, sc in scored_candidates]
        else:
            query_vec = self.encoder.encode_texts([query])
            scores, int_ids = self.index_store.search(query_vec, top_k=search_k)
            q_scores = scores[0]
            q_ids = int_ids[0]

        # 2. Temporal proposals per hit
        raw_moments: list[RankedMoment] = []
        for score, int_id in zip(q_scores, q_ids):
            coord = self.id_mapping.get_by_int_id(int(int_id))
            if coord is None:
                continue

            t_sec = coord.timestamp_seconds if coord.timestamp_seconds is not None else 0.0
            start_sec = max(0.0, t_sec - window_duration_seconds / 2.0)
            end_sec = t_sec + window_duration_seconds / 2.0

            m_id = str(uuid.uuid4())
            moment = RankedMoment(
                moment_id=m_id,
                video_id=coord.video_id,
                moment_start_seconds=start_sec,
                moment_end_seconds=end_sec,
                start_original_frame_index=max(0, coord.ordinal - 15),
                end_original_frame_index=coord.ordinal + 15,
                representative_original_frame_index=coord.ordinal,
                representative_timestamp_seconds=t_sec,
                ranking_score=float(score),
                evidence_frame_indices=(coord.ordinal,),
            )
            raw_moments.append(moment)

        # 4. Temporal NMS within each video
        by_video: dict[str, list[RankedMoment]] = {}
        for m in raw_moments:
            by_video.setdefault(m.video_id, []).append(m)

        pruned_moments: list[RankedMoment] = []
        for vid, v_moments in by_video.items():
            # Sort by descending ranking score
            v_moments.sort(key=lambda m: m.ranking_score, reverse=True)
            kept: list[RankedMoment] = []
            for candidate in v_moments:
                suppress = False
                for existing in kept:
                    iou = temporal_iou(
                        candidate.moment_start_seconds,
                        candidate.moment_end_seconds,
                        existing.moment_start_seconds,
                        existing.moment_end_seconds,
                    )
                    if iou > nms_iou_threshold:
                        suppress = True
                        break
                if not suppress:
                    kept.append(candidate)
            pruned_moments.extend(kept)

        # Global sort across videos by score
        pruned_moments.sort(key=lambda m: m.ranking_score, reverse=True)
        return pruned_moments[:top_k_moments]

    def save(self, directory: Path | str) -> None:
        """Persist index store and deterministic ID mapping into directory."""
        dir_p = Path(directory)
        dir_p.mkdir(parents=True, exist_ok=True)
        self.index_store.save(dir_p / "index")
        self.id_mapping.to_parquet(dir_p / "id_mapping.parquet")

    @classmethod
    def load(
        cls,
        directory: Path | str,
        encoder: SigLIP2Encoder | MockSigLIP2Encoder | None = None,
        prefer_faiss: bool = True,
    ) -> "VisualRetrievalEngine":
        """Load index store and ID mapping from directory."""
        dir_p = Path(directory)
        idx_dir = dir_p / "index"
        map_path = dir_p / "id_mapping.parquet"

        if not map_path.exists():
            raise FileNotFoundError(f"Missing ID mapping at {map_path}")

        index_store = VisualIndexStore.load(idx_dir, prefer_faiss=prefer_faiss)
        id_mapping = IDMappingStore.from_parquet(map_path)
        return cls(encoder=encoder, index_store=index_store, id_mapping=id_mapping)


def compute_keyframe_hubness_radii(
    vectors: np.ndarray,
    video_ids: Sequence[str] | None = None,
    k_neighbors: int = 16,
    sample_anchors: int = 2048,
) -> np.ndarray:
    """Compute cross-video hubness radius r_K(v_i) for each keyframe vector (QAC-CSLS)."""
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    n = vecs.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    rng = np.random.default_rng(42)
    n_anchors = min(n, max(1, int(sample_anchors)))
    anchor_idx = rng.choice(n, size=n_anchors, replace=False)
    anchors = np.asarray(vecs[anchor_idx], dtype=np.float32)
    anchor_norms = np.linalg.norm(anchors, axis=1, keepdims=True)
    anchors = anchors / np.maximum(anchor_norms, 1e-8)

    use_vid_mask = video_ids is not None and len(video_ids) == n
    if use_vid_mask:
        vid_arr = np.asarray(video_ids, dtype=object)
        anchor_vids = vid_arr[anchor_idx]

    k_eff = min(max(1, int(k_neighbors)), anchors.shape[0] - 1)
    if k_eff <= 0:
        return np.zeros(n, dtype=np.float32)

    radii = np.zeros(n, dtype=np.float32)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        chunk = np.asarray(vecs[start:end], dtype=np.float32)
        chunk_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        chunk = chunk / np.maximum(chunk_norms, 1e-8)

        sims = chunk @ anchors.T
        if use_vid_mask:
            same_vid = vid_arr[start:end, None] == anchor_vids[None, :]
            sims = np.where(same_vid, -1.0, sims)

        part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
        # Exclude masked -1.0 entries if a video had very few cross-video anchors
        valid = part > -0.99
        counts = np.sum(valid, axis=1)
        sums = np.sum(np.where(valid, part, 0.0), axis=1)
        radii[start:end] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    return np.nan_to_num(radii, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_or_load_keyframe_hubness_radii(
    vectors: np.ndarray,
    video_ids: Sequence[str] | None = None,
    cache_path: Path | str | None = None,
    k_neighbors: int = 16,
    sample_anchors: int = 2048,
) -> tuple[np.ndarray, float]:
    """Load cached keyframe hubness radii if length matches, else compute and cache."""
    n = len(vectors) if vectors is not None else 0
    if n == 0:
        return np.zeros(0, dtype=np.float32), 0.0

    cp = Path(cache_path) if cache_path is not None else None
    if cp is not None and cp.is_file():
        try:
            loaded = np.load(str(cp))
            if loaded.ndim == 1 and loaded.shape[0] == n and np.isfinite(loaded).all():
                radii = loaded.astype(np.float32, copy=False)
                return radii, float(np.median(radii))
        except Exception:
            pass

    radii = compute_keyframe_hubness_radii(
        vectors,
        video_ids=video_ids,
        k_neighbors=k_neighbors,
        sample_anchors=sample_anchors,
    )
    if cp is not None and len(radii) == n:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cp), radii)
        except Exception:
            pass
    median_r = float(np.median(radii)) if len(radii) > 0 else 0.0
    return radii, median_r


