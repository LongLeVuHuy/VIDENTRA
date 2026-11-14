"""Unified Storage Adapter bridging Blueprint Parquet storage and production SQLite/FAISS corpus."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Sequence
import uuid

import numpy as np

from videntra.retrieval.visual import RankedMoment, ensemble_clause_scores, VisualRetrievalEngine
from videntra.query.rules import decompose_query_clauses, extract_object_requirements
from videntra.indexing.text_store import build_fts5_query
from videntra.relations.entities import compute_object_verification_boost
from videntra.retrieval.fusion import aggregate_video_moments
from videntra.localization.refine import adaptive_temporal_boundary_expansion, smooth_1d_temporal_scores


class UnifiedSearchEngine:
    """Unified facade accommodating both SQLite/FAISS production corpus and Blueprint Parquet stores."""

    def __init__(
        self,
        db_path: Path | str = Path("data/processed/metadata.db"),
        faiss_path: Path | str = Path("data/processed/faiss_index.bin"),
        model_dir: Path | str = Path("models/siglip2"),
        blueprint_store_dir: Path | str | None = None,
        device: str = "cpu",
    ):
        self.db_path = Path(db_path)
        self.faiss_path = Path(faiss_path)
        self.model_dir = Path(model_dir)
        self.device = device
        self.backend = "unknown"

        # 1. Detect SQLite + FAISS production backend
        if self.db_path.exists() and self.faiss_path.exists():
            self._init_sqlite_faiss_backend()
            self.backend = "sqlite_faiss"
        elif blueprint_store_dir and Path(blueprint_store_dir).exists():
            self._init_blueprint_backend(Path(blueprint_store_dir))
            self.backend = "blueprint_parquet"
        else:
            self.backend = "uninitialized"

    def _init_sqlite_faiss_backend(self) -> None:
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.index = faiss.read_index(str(self.faiss_path))
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT faiss_id, video_id, pts_time, source_frame_index, keyframe_path FROM KeyframeMap;")
        self.keyframe_map = {
            row[0]: {
                "video_id": row[1],
                "pts_time": float(row[2]),
                "source_frame_index": row[3],
                "keyframe_path": row[4],
            }
            for row in cursor.fetchall()
        }

        self.video_keyframes: dict[str, list[dict]] = defaultdict(list)
        for fid, kinfo in self.keyframe_map.items():
            self.video_keyframes[kinfo["video_id"]].append({
                "faiss_id": fid,
                "pts_time": kinfo["pts_time"],
                "source_frame_index": kinfo["source_frame_index"],
            })
        for vid in self.video_keyframes:
            self.video_keyframes[vid].sort(key=lambda x: x["pts_time"])

        cursor.execute("SELECT video_id, timestamp_ms, class_name, confidence FROM FrameObjects;")
        self.video_objects: dict[str, list[dict]] = defaultdict(list)
        for vid, ts_ms, cname, conf in cursor.fetchall():
            self.video_objects[vid].append({
                "pts_sec": ts_ms / 1000.0,
                "class_name": cname,
                "confidence": float(conf),
            })

        cursor.execute("SELECT video_id FROM Videos;")
        self.indexed_videos = set(row[0] for row in cursor.fetchall())
        conn.close()

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.text_model = AutoModel.from_pretrained(str(self.model_dir)).to(self.device)
        self.text_model.eval()

        if self.index.ntotal > 0:
            all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
            mean_vec = np.mean(all_vecs, axis=0, keepdims=True)
            mean_vec /= (np.linalg.norm(mean_vec, axis=-1, keepdims=True) + 1e-8)
            self.hubness_bias = (all_vecs @ mean_vec.T).flatten()
            self.all_embeddings = all_vecs
        else:
            self.hubness_bias = np.zeros(0, dtype=np.float32)
            self.all_embeddings = np.zeros((0, 768), dtype=np.float32)

    def _init_blueprint_backend(self, store_dir: Path) -> None:
        self.visual_engine = VisualRetrievalEngine.load(store_dir)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        def extract_features(out):
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state[:, 0, :]
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

        inputs = self.tokenizer(texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(self.text_model, "get_text_features"):
                raw = self.text_model.get_text_features(**inputs)
            else:
                raw = self.text_model(**inputs)
            feats = extract_features(raw)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.cpu().float().numpy()

    def _encode_augmented_query(self, query: str) -> np.ndarray:
        p1 = query
        p2 = f"a video frame showing {query}"
        vecs = self._encode_texts([p1, p2])
        ens = 0.75 * vecs[0] + 0.25 * vecs[1]
        ens /= (np.linalg.norm(ens) + 1e-8)
        return ens.reshape(1, -1)

    def search(self, query: str, top_k: int = 10) -> list[RankedMoment]:
        """Unified search across multimodal corpus."""
        if self.backend == "blueprint_parquet":
            return self.visual_engine.search(query, top_k_moments=top_k)

        if self.backend != "sqlite_faiss":
            return []

        clauses = decompose_query_clauses(query)
        obj_reqs = extract_object_requirements(query)
        fts_query = build_fts5_query(query, enable_prefix=True)

        search_depth = min(250, self.index.ntotal)
        visual_candidates: list[dict] = []

        primary_q_vec = self._encode_augmented_query(query)

        sub_clauses = [c for c in clauses if c.strip().lower() != query.strip().lower()]
        if sub_clauses:
            sub_vecs = self._encode_texts(sub_clauses)
            all_vecs = np.vstack([primary_q_vec, sub_vecs])
            scores_batch, ids_batch = self.index.search(all_vecs, search_depth)

            full_scores = scores_batch[0]
            full_ids = ids_batch[0]
            full_map = dict(zip(full_ids, full_scores))

            clause_maps = [
                dict(zip(ids_batch[i + 1], scores_batch[i + 1]))
                for i in range(len(sub_clauses))
            ]

            cand_faiss_ids = set(full_ids)
            for c_ids in ids_batch[1:]:
                cand_faiss_ids.update(c_ids)

            scored_visual = []
            for fid in cand_faiss_ids:
                if fid not in self.keyframe_map:
                    continue
                s_full = float(full_map.get(fid, 0.0))
                c_scores = [float(cm.get(fid, 0.0)) for cm in clause_maps]
                s_comb = ensemble_clause_scores(full_score=s_full, clause_scores=c_scores)
                scored_visual.append((fid, s_comb))

            scored_visual.sort(key=lambda x: x[1], reverse=True)
            vis_hits = scored_visual[:search_depth]
        else:
            scores_batch, ids_batch = self.index.search(primary_q_vec, search_depth)
            vis_hits = [
                (fid, float(sc))
                for sc, fid in zip(scores_batch[0], ids_batch[0])
                if fid in self.keyframe_map
            ]

        for fid, raw_score in vis_hits:
            info = self.keyframe_map[fid]
            vid = info["video_id"]
            t_sec = info["pts_time"]

            # Dehubbing: subtract 0.08 * hubness_bias
            h_bias = float(self.hubness_bias[fid]) if fid < len(self.hubness_bias) else 0.0
            dehubbed_score = raw_score - 0.08 * h_bias
            final_score = dehubbed_score

            if obj_reqs:
                nearby_objs = [
                    o for o in self.video_objects.get(vid, [])
                    if abs(o["pts_sec"] - t_sec) <= 3.5
                ]
                obj_boost = compute_object_verification_boost(obj_reqs, nearby_objs)
                final_score += 0.04 * obj_boost

            visual_candidates.append({
                "video_id": vid,
                "timestamp_sec": t_sec,
                "score": float(final_score),
                "branch": "visual",
                "faiss_id": fid,
            })

        ocr_candidates: list[dict] = []
        asr_candidates: list[dict] = []

        if fts_query:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT video_id, timestamp_ms, rank FROM fts_ocr WHERE fts_ocr MATCH ? ORDER BY rank LIMIT 40;",
                    (fts_query,),
                )
                for vid, ts_ms, rank in cursor.fetchall():
                    ocr_candidates.append({
                        "video_id": vid,
                        "timestamp_sec": ts_ms / 1000.0,
                        "score": -float(rank),
                        "branch": "ocr",
                    })

                cursor.execute(
                    "SELECT video_id, start_ms, end_ms, rank FROM fts_asr WHERE fts_asr MATCH ? ORDER BY rank LIMIT 40;",
                    (fts_query,),
                )
                for vid, st_ms, et_ms, rank in cursor.fetchall():
                    asr_candidates.append({
                        "video_id": vid,
                        "start_sec": st_ms / 1000.0,
                        "end_sec": et_ms / 1000.0,
                        "timestamp_sec": (st_ms + et_ms) / 2000.0,
                        "score": -float(rank),
                        "branch": "asr",
                    })
            except Exception:
                pass
            finally:
                conn.close()

        k_rrf = 60
        visual_candidates.sort(key=lambda x: x["score"], reverse=True)
        ocr_candidates.sort(key=lambda x: x["score"], reverse=True)
        asr_candidates.sort(key=lambda x: x["score"], reverse=True)

        clusters: list[dict] = []

        # 1. Add Visual Candidates as primary anchors
        for rank, c in enumerate(visual_candidates, 1):
            rrf_val = 1.0 / (k_rrf + rank)
            clusters.append({
                "video_id": c["video_id"],
                "rep_timestamp": c["timestamp_sec"],
                "visual_sim": c["score"],
                "rrf_vis": rrf_val,
                "rrf_ocr": 0.0,
                "rrf_asr": 0.0,
                "branches": {"visual"},
            })

        # 2. Integrate OCR candidates (merge with nearest visual candidate within +/- 3.0s)
        for rank, c in enumerate(ocr_candidates, 1):
            rrf_val = 1.0 / (k_rrf + rank)
            best_cl = None
            best_diff = float("inf")
            for cl in clusters:
                if cl["video_id"] == c["video_id"]:
                    diff = abs(cl["rep_timestamp"] - c["timestamp_sec"])
                    if diff <= 3.0 and diff < best_diff:
                        best_diff = diff
                        best_cl = cl
            if best_cl is not None:
                best_cl["rrf_ocr"] = max(best_cl["rrf_ocr"], rrf_val)
                best_cl["branches"].add("ocr")
            else:
                clusters.append({
                    "video_id": c["video_id"],
                    "rep_timestamp": c["timestamp_sec"],
                    "visual_sim": 0.0,
                    "rrf_vis": 0.0,
                    "rrf_ocr": rrf_val,
                    "rrf_asr": 0.0,
                    "branches": {"ocr"},
                })

        # 3. Integrate ASR candidates (merge with all visual clusters in [start - 2.0s, end + 2.0s])
        for rank, c in enumerate(asr_candidates, 1):
            rrf_val = 1.0 / (k_rrf + rank)
            st = c.get("start_sec", c["timestamp_sec"] - 1.0)
            et = c.get("end_sec", c["timestamp_sec"] + 1.0)
            merged = False
            for cl in clusters:
                if cl["video_id"] == c["video_id"]:
                    if (st - 2.0) <= cl["rep_timestamp"] <= (et + 2.0):
                        cl["rrf_asr"] = max(cl["rrf_asr"], rrf_val)
                        cl["branches"].add("asr")
                        merged = True
            if not merged:
                clusters.append({
                    "video_id": c["video_id"],
                    "rep_timestamp": c["timestamp_sec"],
                    "visual_sim": 0.0,
                    "rrf_vis": 0.0,
                    "rrf_ocr": 0.0,
                    "rrf_asr": rrf_val,
                    "branches": {"asr"},
                })

        # 4. Compute Hybrid Cluster Scores
        raw_moments: list[RankedMoment] = []
        for cl in clusters:
            tot_rrf = cl["rrf_vis"] + 1.5 * cl["rrf_ocr"] + 1.4 * cl["rrf_asr"]
            score = cl["visual_sim"] + 2.5 * tot_rrf
            m = RankedMoment(
                moment_id=f"{cl['video_id']}_{round(cl['rep_timestamp'], 1)}",
                video_id=cl["video_id"],
                moment_start_seconds=max(0.0, cl["rep_timestamp"] - 1.0),
                moment_end_seconds=cl["rep_timestamp"] + 1.0,
                start_original_frame_index=0,
                end_original_frame_index=0,
                representative_original_frame_index=0,
                representative_timestamp_seconds=cl["rep_timestamp"],
                ranking_score=score,
                match_status="+".join(sorted(cl["branches"])),
            )
            raw_moments.append(m)

        aggregated_moments = aggregate_video_moments(raw_moments, top_k=top_k)

        final_ranked: list[RankedMoment] = []
        for moment in aggregated_moments:
            vid = moment.video_id
            rep_t = moment.representative_timestamp_seconds
            kf_list = self.video_keyframes.get(vid, [])

            if len(kf_list) > 1:
                ts_array = [kf["pts_time"] for kf in kf_list]
                best_idx = min(range(len(ts_array)), key=lambda i: abs(ts_array[i] - rep_t))

                kf_fids = [int(kf["faiss_id"]) for kf in kf_list]
                kf_matrix = self.all_embeddings[kf_fids]
                exact_curve = (kf_matrix @ primary_q_vec.T).flatten().tolist()

                # Boost keyframes overlapping OCR or ASR
                for k, kf_t in enumerate(ts_array):
                    for oc in ocr_candidates:
                        if oc["video_id"] == vid and abs(oc["timestamp_sec"] - kf_t) <= 2.0:
                            exact_curve[k] += 0.03
                            break
                    for ac in asr_candidates:
                        if ac["video_id"] == vid:
                            st = ac.get("start_sec", ac["timestamp_sec"] - 1.0)
                            et = ac.get("end_sec", ac["timestamp_sec"] + 1.0)
                            if (st - 1.5) <= kf_t <= (et + 1.5):
                                exact_curve[k] += 0.03
                                break

                start_sec, end_sec = adaptive_temporal_boundary_expansion(
                    keyframe_timestamps=ts_array,
                    keyframe_scores=exact_curve,
                    peak_index=best_idx,
                    min_relative_peak_ratio=0.72,
                    max_gap_seconds=6.0,
                )
            else:
                start_sec = max(0.0, rep_t - 2.5)
                end_sec = rep_t + 2.5

            refined_moment = RankedMoment(
                moment_id=moment.moment_id,
                video_id=vid,
                moment_start_seconds=round(start_sec, 2),
                moment_end_seconds=round(end_sec, 2),
                start_original_frame_index=moment.start_original_frame_index,
                end_original_frame_index=moment.end_original_frame_index,
                representative_original_frame_index=moment.representative_original_frame_index,
                representative_timestamp_seconds=rep_t,
                ranking_score=moment.ranking_score,
                match_status=moment.match_status,
            )
            final_ranked.append(refined_moment)

        return final_ranked
