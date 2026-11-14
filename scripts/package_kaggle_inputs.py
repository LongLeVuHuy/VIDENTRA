"""Package keyframes and benchmark queries for Kaggle GPU embedding inference."""
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq

from videntra.timeline.ledger import exact_seek_frame

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_inputs(
    manifest_path: str = "manifests/benchmark/smoke/benchmark_smoke.media.json",
    ingest_dir: str = "work/benchmark/smoke/ingest",
    media_root: str | None = None,
    output_dir: str = "work/kaggle/inputs",
    max_samples_per_video: int = 100,
):
    if media_root is None:
        media_root = os.environ.get("VIDENTRA_QVH_MEDIA_DIR", "data/raw/qvhighlights/media")
    out_p = Path(output_dir)
    images_p = out_p / "images"
    images_p.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    ingest_p = Path(ingest_dir)
    media_root_p = Path(media_root)

    keyframe_records = []
    print("Extracting keyframes for Kaggle packaging...")

    for rec in manifest_doc["records"]:
        vid = rec["video_id"]
        sha = rec["sha256"]
        media_name = Path(rec["media_uri"]).name
        media_path = media_root_p / media_name
        if not media_path.exists():
            continue

        tl_matches = list(ingest_p.glob(f"timeline/*{sha}*/*/frames.parquet"))
        samples_matches = list(ingest_p.glob(f"samples/*{sha}*/*/samples.parquet"))
        if not tl_matches or not samples_matches:
            continue

        tl_path = tl_matches[0]
        samples_table = pq.read_table(str(samples_matches[0]))

        ords = samples_table.column("original_decoded_frame_index").to_pylist()
        s_ids = samples_table.column("sampled_frame_id").to_pylist()
        ts_secs = samples_table.column("timestamp_seconds").to_pylist()

        # Sample frames up to max_samples_per_video with uniform stride
        stride = max(1, len(ords) // max_samples_per_video)
        selected_indices = list(range(0, len(ords), stride))[:max_samples_per_video]

        for idx in selected_indices:
            ord_val = ords[idx]
            s_id = s_ids[idx]
            t_sec = ts_secs[idx]

            img_filename = f"{s_id}.jpg"
            img_path = images_p / img_filename

            if not img_path.exists():
                try:
                    frame = exact_seek_frame(media_path, tl_path, target_ordinal=ord_val)
                    if frame.pixels is not None:
                        pil_img = Image.fromarray(frame.pixels)
                        # Resize to standard SigLIP 2 224x224
                        pil_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
                        pil_img.save(img_path, format="JPEG", quality=90)
                except Exception as e:
                    continue

            if img_path.exists():
                keyframe_records.append({
                    "sampled_frame_id": s_id,
                    "video_id": vid,
                    "source_video_name": media_name,
                    "ordinal": ord_val,
                    "timestamp_seconds": round(float(t_sec), 4) if t_sec is not None else 0.0,
                    "image_file": img_filename,
                })

    # Save keyframes manifest
    kf_manifest_path = out_p / "keyframes_manifest.jsonl"
    with open(kf_manifest_path, "w", encoding="utf-8") as f:
        for r in keyframe_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Define stratified benchmark queries
    benchmark_queries = [
        # Visual-dominant
        {"qid": "Q_VIS_01", "query": "A person makes a gift package while sitting at a table", "modality_focus": "visual"},
        {"qid": "Q_VIS_02", "query": "Asian woman holds up beauty products to the camera", "modality_focus": "visual"},
        {"qid": "Q_VIS_03", "query": "A group of men is walking in a deserted path", "modality_focus": "visual"},
        {"qid": "Q_VIS_04", "query": "Mike Pence is sitting in front of a blue background", "modality_focus": "visual"},
        # OCR-dominant
        {"qid": "Q_OCR_01", "query": "VIDENTRA 2026", "modality_focus": "ocr"},
        # ASR-dominant
        {"qid": "Q_ASR_01", "query": "thông lệ không một người đàn ông", "modality_focus": "asr"},
        {"qid": "Q_ASR_02", "query": "nguyên liệu nhảy khiến chị thật thú", "modality_focus": "asr"},
        # Multimodal
        {"qid": "Q_MM_01", "query": "Asian woman holds up beauty products and talks about ingredients", "modality_focus": "multimodal"},
        {"qid": "Q_MM_02", "query": "A person makes a gift package with sign VIDENTRA 2026", "modality_focus": "multimodal"},
    ]

    queries_path = out_p / "benchmark_queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in benchmark_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Save manifest with checksums
    package_summary = {
        "num_keyframes": len(keyframe_records),
        "num_queries": len(benchmark_queries),
        "keyframes_manifest_sha256": sha256_file(kf_manifest_path),
        "queries_sha256": sha256_file(queries_path),
    }

    with open(out_p / "package_manifest.json", "w", encoding="utf-8") as f:
        json.dump(package_summary, f, indent=2)

    # Separately create Ground Truth for Evaluator (strictly isolated from Kaggle inputs)
    gt_dir = Path("work/benchmark/smoke")
    gt_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = [
        {
            "qid": "Q_VIS_01",
            "relevant_videos": [{"video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a", "relevant_windows": [[96.0, 126.0]]}],
        },
        {
            "qid": "Q_VIS_02",
            "relevant_videos": [{"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 22.0]]}],
        },
        {
            "qid": "Q_VIS_03",
            "relevant_videos": [{"video_id": "sha256:724761572dc8714aa2b8083239469cfa6b83da3988e8903c2e13871f94f0d766", "relevant_windows": [[48.0, 68.0]]}],
        },
        {
            "qid": "Q_VIS_04",
            "relevant_videos": [{"video_id": "sha256:0985cda8436a32a31e6ec063a58dc5f30daa937dd8a18bb4798d3347523780e7", "relevant_windows": [[6.0, 14.0], [16.0, 24.0], [36.0, 48.0], [72.0, 82.0], [96.0, 100.0], [108.0, 112.0], [118.0, 124.0], [136.0, 144.0]]}],
        },
        {
            "qid": "Q_OCR_01",
            "relevant_videos": [
                {"video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a", "relevant_windows": [[0.0, 150.0]]},
                {"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 150.0]]},
                {"video_id": "sha256:724761572dc8714aa2b8083239469cfa6b83da3988e8903c2e13871f94f0d766", "relevant_windows": [[0.0, 150.0]]},
                {"video_id": "sha256:0985cda8436a32a31e6ec063a58dc5f30daa937dd8a18bb4798d3347523780e7", "relevant_windows": [[0.0, 150.0]]},
            ],
        },
        {
            "qid": "Q_ASR_01",
            "relevant_videos": [{"video_id": "sha256:0985cda8436a32a31e6ec063a58dc5f30daa937dd8a18bb4798d3347523780e7", "relevant_windows": [[0.0, 30.0]]}],
        },
        {
            "qid": "Q_ASR_02",
            "relevant_videos": [{"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 30.0]]}],
        },
        {
            "qid": "Q_MM_01",
            "relevant_videos": [{"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 22.0]]}],
        },
        {
            "qid": "Q_MM_02",
            "relevant_videos": [{"video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a", "relevant_windows": [[96.0, 126.0]]}],
        },
    ]
    gt_file = gt_dir / "ground_truth.jsonl"
    with open(gt_file, "w", encoding="utf-8") as f:
        for item in ground_truth:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Packaged {len(keyframe_records)} keyframes and {len(benchmark_queries)} queries in {out_p}")
    print(f"Ground truth isolated and saved to {gt_file}")
    return package_summary

if __name__ == "__main__":
    package_inputs()
