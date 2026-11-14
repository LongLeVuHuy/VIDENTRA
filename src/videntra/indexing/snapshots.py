"""Consistent snapshot manifests ensuring query-time atomic index isolation."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from videntra.registry.store import sha256_file


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    created_at_utc: str
    visual_index_dir: str
    text_index_path: str
    evidence_dir: str
    video_ids: tuple[str, ...]
    content_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> "SnapshotManifest":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Missing snapshot manifest at {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            snapshot_id=data["snapshot_id"],
            created_at_utc=data["created_at_utc"],
            visual_index_dir=data["visual_index_dir"],
            text_index_path=data["text_index_path"],
            evidence_dir=data["evidence_dir"],
            video_ids=tuple(data.get("video_ids", ())),
            content_hashes=data.get("content_hashes", {}),
        )


def build_snapshot_manifest(
    snapshot_id: str,
    visual_index_dir: Path | str,
    text_index_path: Path | str,
    evidence_dir: Path | str,
    video_ids: Sequence[str] = (),
) -> SnapshotManifest:
    """Build immutable snapshot record with sha256 hashes of constituent indices."""
    v_dir = Path(visual_index_dir)
    t_path = Path(text_index_path)
    ev_dir = Path(evidence_dir)

    hashes = {}
    if t_path.is_file():
        hashes["text_index"] = sha256_file(t_path)

    id_map = v_dir / "id_mapping.parquet"
    if id_map.is_file():
        hashes["visual_id_mapping"] = sha256_file(id_map)
    for name, key in (("index.faiss", "visual_index_faiss"), ("index_numpy.npz", "visual_index_numpy")):
        index_file = v_dir / "index" / name
        if index_file.is_file():
            hashes[key] = sha256_file(index_file)

    now = datetime.now(timezone.utc).isoformat()
    return SnapshotManifest(
        snapshot_id=snapshot_id,
        created_at_utc=now,
        visual_index_dir=str(v_dir),
        text_index_path=str(t_path),
        evidence_dir=str(ev_dir),
        video_ids=tuple(sorted(video_ids)),
        content_hashes=hashes,
    )


def verify_snapshot_artifacts(manifest: SnapshotManifest) -> bool:
    """Verify recorded bytes, not model compatibility or media rights."""
    visual = Path(manifest.visual_index_dir)
    paths = {"text_index": Path(manifest.text_index_path),
             "visual_id_mapping": visual / "id_mapping.parquet",
             "visual_index_faiss": visual / "index" / "index.faiss",
             "visual_index_numpy": visual / "index" / "index_numpy.npz"}
    hashes = manifest.content_hashes
    if "visual_id_mapping" not in hashes or not ({"visual_index_faiss", "visual_index_numpy"} & hashes.keys()):
        return False
    if any(path.is_file() and key not in hashes for key, path in paths.items()):
        return False
    return all(key in paths and paths[key].is_file() and sha256_file(paths[key]) == value
               for key, value in hashes.items())
