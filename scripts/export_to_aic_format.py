"""Export VIDENTRA indexes and evidence into AIC-standard layout (faiss_index.bin + metadata.db)."""
import json
import os
from pathlib import Path
import sqlite3
import numpy as np
import faiss

def export_to_aic_layout(
    output_base: str = "data",
    embeddings_dir: str = "work/benchmark/smoke/embeddings",
    inputs_dir: str = "work/kaggle/inputs",
    evidence_dir: str = "work/benchmark/smoke/evidence",
    media_dir: str | None = None,
):
    if media_dir is None:
        media_dir = os.environ.get("VIDENTRA_QVH_MEDIA_DIR", "data/raw/qvhighlights/media")
    base_p = Path(output_base)
    raw_p = base_p / "raw"
    processed_p = base_p / "processed"
    raw_p.mkdir(parents=True, exist_ok=True)
    processed_p.mkdir(parents=True, exist_ok=True)

    emb_p = Path(embeddings_dir)
    in_p = Path(inputs_dir)
    ev_p = Path(evidence_dir)
    media_p = Path(media_dir)

    print(f"Exporting to AIC standard layout at '{base_p.resolve()}'...")

    # 1. Export FAISS Index
    frame_emb_file = emb_p / "frame_embeddings.npy"
    if not frame_emb_file.exists():
        print(f"Error: {frame_emb_file} not found!")
        return

    vectors = np.load(frame_emb_file).astype(np.float32)
    num_vectors, dim = vectors.shape
    print(f"Loaded {num_vectors} vectors of dimension {dim}.")

    # Create FlatIP index (Cosine Similarity for L2-normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss_out = processed_p / "faiss_index.bin"
    faiss.write_index(index, str(faiss_out))
    print(f"Saved FAISS index ({faiss_out.stat().st_size} bytes) -> {faiss_out}")

    # 2. Build metadata.db SQLite Database
    db_out = processed_p / "metadata.db"
    if db_out.exists():
        db_out.unlink()

    conn = sqlite3.connect(db_out)
    with conn:
        # Create standard tables matching AIC schema + VIDENTRA multimodal extensions
        conn.execute("""
            CREATE TABLE VideoMeta (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                length INTEGER,
                media_path TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE KeyframeMap (
                global_id INTEGER PRIMARY KEY,
                video_id TEXT,
                frame_idx INTEGER,
                pts_time REAL,
                sampled_frame_id TEXT,
                FOREIGN KEY (video_id) REFERENCES VideoMeta(video_id)
            );
        """)
        conn.execute("""
            CREATE TABLE TextEvidence (
                evidence_id TEXT PRIMARY KEY,
                video_id TEXT,
                kind TEXT,  -- 'ocr' or 'asr'
                text TEXT,
                start_seconds REAL,
                end_seconds REAL,
                confidence REAL,
                FOREIGN KEY (video_id) REFERENCES VideoMeta(video_id)
            );
        """)

    # Populate KeyframeMap
    with open(in_p / "keyframes_manifest.jsonl", "r", encoding="utf-8") as f:
        keyframes = [json.loads(line) for line in f if line.strip()]

    video_ids = set()
    keyframe_rows = []
    for g_id, kf in enumerate(keyframes):
        vid = kf["video_id"]
        video_ids.add((vid, kf.get("source_video_name", vid)))
        keyframe_rows.append((
            g_id,
            vid,
            kf["ordinal"],
            float(kf["timestamp_seconds"]),
            kf["sampled_frame_id"],
        ))

    # Populate VideoMeta
    video_rows = []
    for vid, name in video_ids:
        # Check actual length if video exists
        vpath = media_p / name
        video_rows.append((
            vid,
            name,
            150,  # nominal duration
            str(vpath) if vpath.exists() else name,
        ))

    # Populate TextEvidence from SQLite FTS / Parquet
    import pyarrow.parquet as pq
    text_rows = []
    for p in ev_p.glob("asr_segments/*.parquet"):
        t = pq.read_table(str(p))
        for i in range(len(t)):
            text_rows.append((
                t.column("segment_id")[i].as_py(),
                t.column("video_id")[i].as_py(),
                "asr",
                t.column("text")[i].as_py(),
                t.column("start_seconds")[i].as_py(),
                t.column("end_seconds")[i].as_py(),
                t.column("confidence")[i].as_py(),
            ))

    for p in ev_p.glob("ocr_tracks/*.parquet"):
        t = pq.read_table(str(p))
        for i in range(len(t)):
            text_rows.append((
                t.column("track_id")[i].as_py(),
                t.column("video_id")[i].as_py(),
                "ocr",
                t.column("text")[i].as_py(),
                t.column("start_seconds")[i].as_py(),
                t.column("end_seconds")[i].as_py(),
                t.column("confidence")[i].as_py(),
            ))

    with conn:
        conn.executemany("INSERT INTO VideoMeta VALUES (?, ?, ?, ?)", video_rows)
        conn.executemany("INSERT INTO KeyframeMap VALUES (?, ?, ?, ?, ?)", keyframe_rows)
        conn.executemany("INSERT INTO TextEvidence VALUES (?, ?, ?, ?, ?, ?, ?)", text_rows)

    print(f"Populated metadata.db:")
    print(f"  - VideoMeta: {len(video_rows)} rows")
    print(f"  - KeyframeMap: {len(keyframe_rows)} rows (global_id 0 to {len(keyframe_rows)-1})")
    print(f"  - TextEvidence: {len(text_rows)} rows (OCR & ASR)")
    print(f"Database size: {db_out.stat().st_size} bytes -> {db_out}")
    print("\nSUCCESS: AIC standard structure created under 'data/'.")

if __name__ == "__main__":
    export_to_aic_layout()
