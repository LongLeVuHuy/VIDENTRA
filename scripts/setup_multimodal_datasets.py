"""Setup and register Multimodal Video Search Datasets (MSR-VTT 1kA, TextVR, TVR, QVHighlights)."""
import json
import os
from pathlib import Path
import urllib.request
import zipfile
from huggingface_hub import hf_hub_download

ASSET_ROOT = Path(os.environ.get("VIDENTRA_ASSETS_ROOT", "data/raw"))

def setup_msrvtt():
    print("=== Setting up MSR-VTT 1kA Multimodal Dataset ===")
    msrvtt_dir = ASSET_ROOT / "msrvtt"
    media_dir = msrvtt_dir / "media"
    prov_dir = msrvtt_dir / "provenance"
    media_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    zip_path = msrvtt_dir / "msrvtt-1ka-original-videos-v1.zip"
    if zip_path.exists():
        print(f"Extracting videos from {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                if member.filename.endswith(".mp4"):
                    # Flatten filename to media_dir / videoXXXX.mp4
                    fname = Path(member.filename).name
                    target = media_dir / fname
                    if not target.exists():
                        with z.open(member) as source, open(target, "wb") as dest:
                            dest.write(source.read())
        print(f"Extraction done. Total videos in {media_dir}: {len(list(media_dir.glob('*.mp4')))}")
        # Delete zip to save disk space
        zip_path.unlink()
        print("Removed temporary zip archive.")

    # Download MSR-VTT test annotations
    ann_target = prov_dir / "test_videodatainfo.json"
    if not ann_target.exists():
        print("Downloading MSR-VTT test queries from HuggingFace (AlexZigma/msr-vtt)...")
        ann_zip = hf_hub_download(
            repo_id="AlexZigma/msr-vtt",
            filename="data/test_videodatainfo.json.zip",
            repo_type="dataset",
        )
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(prov_dir)
        print("MSR-VTT test queries downloaded and extracted.")

def setup_textvr():
    print("\n=== Setting up TextVR (Scene-Text Video Retrieval) ===")
    textvr_dir = ASSET_ROOT / "textvr"
    prov_dir = textvr_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    ann_target = prov_dir / "TextVR_test.json"
    if not ann_target.exists():
        url = "https://raw.githubusercontent.com/callsys/TextVR/main/data/TextVR/TextVR_test.json"
        print(f"Downloading TextVR test annotations from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(ann_target, "wb") as f:
            f.write(resp.read())
        print("TextVR test annotations downloaded successfully.")

def setup_tvr():
    print("\n=== Setting up TVR (TV show Video-Subtitle Retrieval) ===")
    tvr_dir = ASSET_ROOT / "tvr"
    prov_dir = tvr_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    ann_target = prov_dir / "tvr_val_release.jsonl"
    if not ann_target.exists():
        url = "https://raw.githubusercontent.com/jayleicn/TVRetrieval/master/data/tvr_val_release.jsonl"
        print(f"Downloading TVR val annotations from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(ann_target, "wb") as f:
            f.write(resp.read())
        print("TVR validation annotations downloaded successfully.")

def verify_inventory():
    print("\n================ DATASET INVENTORY SUMMARY ================")
    # 1. QVHighlights
    qvh_media = list((ASSET_ROOT / "qvhighlights/media").glob("*.mp4"))
    print(f"1. QVHighlights: {len(qvh_media)} videos (MP4 with audio streams)")
    print("   Modalities: Visual + Audio/Speech + Scene-Text")
    print("   Annotations: highlight_val_release.jsonl (1,550 moment queries)")

    # 2. MSR-VTT
    msrvtt_media = list((ASSET_ROOT / "msrvtt/media").glob("*.mp4"))
    msrvtt_ann = ASSET_ROOT / "msrvtt/provenance/test_videodatainfo.json"
    msrvtt_queries = 0
    if msrvtt_ann.exists():
        with open(msrvtt_ann, "r", encoding="utf-8") as f:
            data = json.load(f)
            msrvtt_queries = len(data.get("sentences", []))
    print(f"\n2. MSR-VTT 1kA: {len(msrvtt_media)} videos")
    print("   Modalities: Visual actions + Audio/Speech + Scene text + Natural Language Queries")
    print(f"   Annotations: {msrvtt_queries} multi-sentence test queries")

    # 3. TextVR
    textvr_ann = ASSET_ROOT / "textvr/provenance/TextVR_test.json"
    textvr_queries = 0
    if textvr_ann.exists():
        with open(textvr_ann, "r", encoding="utf-8") as f:
            tvr_data = json.load(f)
            textvr_queries = len(tvr_data)
    print(f"\n3. TextVR (Reading Comprehension in Video Retrieval):")
    print(f"   Annotations: {textvr_queries} scene-text targeted natural language queries")
    print("   Modalities: Cross-modal Visual + High-density Scene-Text (OCR)")

    # 4. TVR
    tvr_ann = ASSET_ROOT / "tvr/provenance/tvr_val_release.jsonl"
    tvr_queries = 0
    if tvr_ann.exists():
        with open(tvr_ann, "r", encoding="utf-8") as f:
            tvr_queries = sum(1 for line in f if line.strip())
    print(f"\n4. TVR (TV show Video-Subtitle Retrieval):")
    print(f"   Annotations: {tvr_queries} moment queries across 6 TV shows")
    print("   Modalities: Visual + Dialogue/Subtitles/Speech (ASR) + Temporal Window Labels")

if __name__ == "__main__":
    setup_msrvtt()
    setup_textvr()
    setup_tvr()
    verify_inventory()
