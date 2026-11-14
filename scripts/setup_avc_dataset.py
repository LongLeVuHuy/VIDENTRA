"""Setup and register AVC (Aligning Video Corpus) multimodal search dataset."""
import json
import os
from pathlib import Path
import zipfile

AVC_ROOT = Path(os.environ.get("VIDENTRA_ASSETS_ROOT", "data/raw")) / "avc"
MEDIA_DIR = AVC_ROOT / "media"
PROV_DIR = AVC_ROOT / "provenance"

def setup_avc():
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    PROV_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract AVC 300 video bundle
    bundle_zip = AVC_ROOT / "phase8-avc-pilot300-bundle-20261102.zip"
    if bundle_zip.exists():
        print(f"Extracting AVC 300 videos from {bundle_zip.name}...")
        with zipfile.ZipFile(bundle_zip, "r") as z:
            for member in z.infolist():
                if member.filename.endswith(".mp4"):
                    fname = Path(member.filename).name
                    target = MEDIA_DIR / fname
                    if not target.exists():
                        with z.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
        print(f"Extraction done. Total AVC videos in {MEDIA_DIR}: {len(list(MEDIA_DIR.glob('*.mp4')))}")
        bundle_zip.unlink()
        print("Removed bundle zip archive.")

    # 2. Extract policy inputs and query candidates
    for p_zip in PROV_DIR.glob("*.zip"):
        print(f"Extracting {p_zip.name}...")
        with zipfile.ZipFile(p_zip, "r") as z:
            z.extractall(PROV_DIR)
        p_zip.unlink()
        print(f"Extracted and removed {p_zip.name}.")

    # 3. Update asset roots
    config_file = Path("configs/local/asset_roots.json")
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["roots"]["avc"] = str(MEDIA_DIR)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("Registered 'avc' in configs/local/asset_roots.json.")

    # 4. Inventory check
    videos = list(MEDIA_DIR.glob("*.mp4"))
    prov_files = list(PROV_DIR.rglob("*.*"))
    print("\n================ AVC DATASET INVENTORY ================")
    print(f"Total AVC Videos: {len(videos)} clips")
    print(f"Provenance / Query / Policy Files: {len(prov_files)} files")
    for pf in prov_files[:5]:
        print(f"  - {pf.name} ({pf.stat().st_size} bytes)")

if __name__ == "__main__":
    setup_avc()
