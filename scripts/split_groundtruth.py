"""Split Ground Truth annotations into validation (80%) and test_queries (20%) sets.

Maintains exact dataset schemas and isolates validation and test sets per dataset.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def split_list(items: list, ratio: float = 0.8, seed: int = 42) -> tuple[list, list]:
    """Deterministically splits a list into two subsets."""
    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_idx = int(len(items) * ratio)
    val_indices = set(indices[:split_idx])

    val_items = [items[i] for i in range(len(items)) if i in val_indices]
    test_items = [items[i] for i in range(len(items)) if i not in val_indices]
    return val_items, test_items


def process_splits(
    raw_dir: Path = Path("data/raw"),
    val_dir: Path = Path("validation"),
    test_queries_dir: Path = Path("test_queries"),
    ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    print("=" * 75)
    print("PHÂN CHIA TẬP GROUNDTRUTH: 80% VALIDATION & 20% TEST_QUERIES")
    print("=" * 75)

    stats = {}

    # 1. AVC
    print("\n--- 1. Xử lý dataset AVC ---")
    avc_gt = raw_dir / "avc" / "GROUNDTRUTH"
    avc_val_dir = val_dir / "avc"
    avc_test_dir = test_queries_dir / "avc"
    avc_val_dir.mkdir(parents=True, exist_ok=True)
    avc_test_dir.mkdir(parents=True, exist_ok=True)

    avc_plan = avc_gt / "candidate_plan_inference.json"
    if avc_plan.exists():
        with open(avc_plan, "r", encoding="utf-8") as f:
            avc_data = json.load(f)

        queries = avc_data.get("queries", [])
        val_q, test_q = split_list(queries, ratio, seed)

        val_data = copy.deepcopy(avc_data)
        val_data["queries"] = val_q
        with open(avc_val_dir / "candidate_plan_inference.json", "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        test_data = copy.deepcopy(avc_data)
        test_data["queries"] = test_q
        with open(avc_test_dir / "candidate_plan_inference.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        # Copy auxiliary marker if exists
        marker = avc_gt / "source_bundle_marker.json"
        if marker.exists():
            shutil.copy2(marker, avc_val_dir / "source_bundle_marker.json")
            shutil.copy2(marker, avc_test_dir / "source_bundle_marker.json")

        stats["avc"] = {"total_queries": len(queries), "val_queries": len(val_q), "test_queries": len(test_q)}
        print(f"  AVC Queries: Tổng={len(queries)} -> Val={len(val_q)} (80%) | Test_queries={len(test_q)} (20%)")

    # 2. MSR-VTT
    print("\n--- 2. Xử lý dataset MSR-VTT ---")
    msr_gt = raw_dir / "msrvtt" / "GROUNDTRUTH"
    msr_val_dir = val_dir / "msrvtt"
    msr_test_dir = test_queries_dir / "msrvtt"
    msr_val_dir.mkdir(parents=True, exist_ok=True)
    msr_test_dir.mkdir(parents=True, exist_ok=True)

    msr_file = msr_gt / "test_videodatainfo.json"
    if msr_file.exists():
        with open(msr_file, "r", encoding="utf-8") as f:
            msr_data = json.load(f)

        videos = msr_data.get("videos", [])
        sentences = msr_data.get("sentences", [])

        # Split by video IDs to avoid data leakage
        val_vids, test_vids = split_list(videos, ratio, seed)
        val_vid_set = {v["video_id"] for v in val_vids}

        val_sentences = [s for s in sentences if s["video_id"] in val_vid_set]
        test_sentences = [s for s in sentences if s["video_id"] not in val_vid_set]

        with open(msr_val_dir / "test_videodatainfo.json", "w", encoding="utf-8") as f:
            json.dump({
                "info": msr_data.get("info", {}),
                "videos": val_vids,
                "sentences": val_sentences,
            }, f, indent=2, ensure_ascii=False)

        with open(msr_test_dir / "test_videodatainfo.json", "w", encoding="utf-8") as f:
            json.dump({
                "info": msr_data.get("info", {}),
                "videos": test_vids,
                "sentences": test_sentences,
            }, f, indent=2, ensure_ascii=False)

        stats["msrvtt"] = {
            "total_videos": len(videos),
            "val_videos": len(val_vids),
            "test_videos": len(test_vids),
            "total_sentences": len(sentences),
            "val_sentences": len(val_sentences),
            "test_sentences": len(test_sentences),
        }
        print(f"  MSR-VTT Videos: Tổng={len(videos)} -> Val={len(val_vids)} | Test_queries={len(test_vids)}")
        print(f"  MSR-VTT Queries: Tổng={len(sentences)} -> Val={len(val_sentences)} | Test_queries={len(test_sentences)}")

    # 3. QVHighlights
    print("\n--- 3. Xử lý dataset QVHighlights ---")
    qvh_gt = raw_dir / "qvhighlights" / "GROUNDTRUTH"
    qvh_val_dir = val_dir / "qvhighlights"
    qvh_test_dir = test_queries_dir / "qvhighlights"
    qvh_val_dir.mkdir(parents=True, exist_ok=True)
    qvh_test_dir.mkdir(parents=True, exist_ok=True)

    qvh_file = qvh_gt / "highlight_val_release.jsonl"
    if qvh_file.exists():
        with open(qvh_file, "r", encoding="utf-8") as f:
            qvh_lines = [line.strip() for line in f if line.strip()]

        val_lines, test_lines = split_list(qvh_lines, ratio, seed)

        with open(qvh_val_dir / "highlight_val_release.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(val_lines) + "\n")

        with open(qvh_test_dir / "highlight_val_release.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(test_lines) + "\n")

        # Copy license & readme
        for aux in ("ANNOTATIONS_LICENSE.txt", "data_README.md"):
            aux_p = qvh_gt / aux
            if aux_p.exists():
                shutil.copy2(aux_p, qvh_val_dir / aux)
                shutil.copy2(aux_p, qvh_test_dir / aux)

        stats["qvhighlights"] = {"total_queries": len(qvh_lines), "val_queries": len(val_lines), "test_queries": len(test_lines)}
        print(f"  QVHighlights Queries: Tổng={len(qvh_lines)} -> Val={len(val_lines)} | Test_queries={len(test_lines)}")

    # 4. TextVR
    print("\n--- 4. Xử lý dataset TextVR ---")
    tvr_txt_gt = raw_dir / "textvr" / "GROUNDTRUTH"
    textvr_val_dir = val_dir / "textvr"
    textvr_test_dir = test_queries_dir / "textvr"
    textvr_val_dir.mkdir(parents=True, exist_ok=True)
    textvr_test_dir.mkdir(parents=True, exist_ok=True)

    textvr_file = tvr_txt_gt / "TextVR_test.json"
    if textvr_file.exists():
        with open(textvr_file, "r", encoding="utf-8") as f:
            textvr_data = json.load(f)

        val_records, test_records = split_list(textvr_data, ratio, seed)

        with open(textvr_val_dir / "TextVR_test.json", "w", encoding="utf-8") as f:
            json.dump(val_records, f, indent=2, ensure_ascii=False)

        with open(textvr_test_dir / "TextVR_test.json", "w", encoding="utf-8") as f:
            json.dump(test_records, f, indent=2, ensure_ascii=False)

        stats["textvr"] = {"total_records": len(textvr_data), "val_records": len(val_records), "test_records": len(test_records)}
        print(f"  TextVR Records: Tổng={len(textvr_data)} -> Val={len(val_records)} | Test_queries={len(test_records)}")

    # 5. TVR
    print("\n--- 5. Xử lý dataset TVR ---")
    tvr_gt = raw_dir / "tvr" / "GROUNDTRUTH"
    tvr_val_dir = val_dir / "tvr"
    tvr_test_dir = test_queries_dir / "tvr"
    tvr_val_dir.mkdir(parents=True, exist_ok=True)
    tvr_test_dir.mkdir(parents=True, exist_ok=True)

    tvr_file = tvr_gt / "tvr_val_release.jsonl"
    if tvr_file.exists():
        with open(tvr_file, "r", encoding="utf-8") as f:
            tvr_lines = [line.strip() for line in f if line.strip()]

        val_lines, test_lines = split_list(tvr_lines, ratio, seed)

        with open(tvr_val_dir / "tvr_val_release.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(val_lines) + "\n")

        with open(tvr_test_dir / "tvr_val_release.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(test_lines) + "\n")

        stats["tvr"] = {"total_queries": len(tvr_lines), "val_queries": len(val_lines), "test_queries": len(test_lines)}
        print(f"  TVR Queries: Tổng={len(tvr_lines)} -> Val={len(val_lines)} | Test_queries={len(test_lines)}")

    print(f"\n{'='*75}")
    print("HOÀN THÀNH PHÂN CHIA TOÀN BỘ 5 TẬP GROUNDTRUTH")
    print(f"{'='*75}")
    print(f"- Thư mục validation   : {val_dir.resolve()}")
    print(f"- Thư mục test_queries : {test_queries_dir.resolve()}")
    print(f"{'='*75}\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--test-queries-dir", type=Path, default=Path("test_queries"))
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_splits(
        raw_dir=args.raw_dir,
        val_dir=args.val_dir,
        test_queries_dir=args.test_queries_dir,
        ratio=args.ratio,
        seed=args.seed,
    )

