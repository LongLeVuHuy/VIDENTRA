"""CPU-only deterministic 40/40/20 source-disjoint split lock for ActivityNet Captions & VideoComp."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_SPLIT_SEED = 20261124
_CLIP_SUFFIX_RE = re.compile(r"(?:_seg\d+.*|_clip\d+.*|#\d+)$", flags=re.IGNORECASE)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def normalize_source_key(raw_id: str) -> str:
    """Normalize ActivityNet/YouCook2 raw video ID (`v_xyz`, cut suffixes) to root clip identity."""
    vid = str(raw_id or "").strip()
    if vid.startswith("v_") and len(vid) > 2:
        vid = vid[2:]
    vid = _CLIP_SUFFIX_RE.sub("", vid)
    return vid.strip()


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra <= rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _deterministic_group_rank(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{group_id}".encode("utf-8")).hexdigest()


def build_source_group_catalog(
    records: Sequence[Mapping[str, Any]],
    *,
    file_sha256_by_video_id: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Group media across Captions + VideoComp (`source_video_id`, file hash, duplicate clusters).

    Returns `(clean_groups_by_id, quarantined_records)` where cross-pool overlaps (`train` + `validation`)
    or unknown-source records are separated prior to split lock.
    """
    all_keys: list[str] = []
    rec_root_keys: list[str] = []
    for idx, rec in enumerate(records):
        vid = str(rec.get("video_id") or f"row_{idx}")
        explicit_src = str(rec.get("source_group_id") or rec.get("source_video_id") or "").strip()
        root = normalize_source_key(explicit_src or vid)
        rec_root_keys.append(root)
        all_keys.append(root)

    uf = _UnionFind(set(all_keys))
    if file_sha256_by_video_id:
        roots_by_hash: dict[str, list[str]] = defaultdict(list)
        for rec, root in zip(records, rec_root_keys):
            vid = str(rec.get("video_id") or "")
            fsha = file_sha256_by_video_id.get(vid) or file_sha256_by_video_id.get(root)
            if fsha:
                roots_by_hash[str(fsha)].append(root)
        for shared_roots in roots_by_hash.values():
            first = shared_roots[0]
            for other in shared_roots[1:]:
                uf.union(first, other)

    grouped_recs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pools_by_group: dict[str, set[str]] = defaultdict(set)
    domains_by_group: dict[str, set[str]] = defaultdict(set)
    video_ids_by_group: dict[str, set[str]] = defaultdict(set)

    for rec, root in zip(records, rec_root_keys):
        gid = uf.find(root)
        grouped_recs[gid].append(rec)
        pool = str(rec.get("public_pool") or "unknown").strip().lower()
        dom = str(rec.get("domain") or "activitynet").strip().lower()
        vid = str(rec.get("video_id") or root)
        pools_by_group[gid].add(pool)
        domains_by_group[gid].add(dom)
        video_ids_by_group[gid].add(vid)

    clean_groups: dict[str, dict[str, Any]] = {}
    quarantined: list[dict[str, Any]] = []
    for gid in sorted(grouped_recs):
        pools = pools_by_group[gid]
        if "unknown" in pools or len(pools) > 1 or not (pools <= {"train", "validation"}):
            reason = "cross_pool_overlap" if len(pools) > 1 else "unknown_or_invalid_pool"
            quarantined.append({
                "source_group_id": gid,
                "pools": sorted(pools),
                "domains": sorted(domains_by_group[gid]),
                "video_ids": sorted(video_ids_by_group[gid]),
                "reason": reason,
                "record_count": len(grouped_recs[gid]),
            })
            continue
        clean_groups[gid] = {
            "source_group_id": gid,
            "public_pool": next(iter(pools)),
            "domains": sorted(domains_by_group[gid]),
            "video_ids": sorted(video_ids_by_group[gid]),
            "records": grouped_recs[gid],
        }
    return clean_groups, quarantined


def assign_source_disjoint_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    max_groups_per_domain: int | None = None,
    annotation_version: str = "v1",
    file_sha256_by_video_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Deterministically assign 40% Tune / 40% Valid-confirm (from train) & 20% Test (from val) zero-overlap."""
    clean_groups, quarantined = build_source_group_catalog(
        records, file_sha256_by_video_id=file_sha256_by_video_id
    )

    by_domain_pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    all_domains: set[str] = set()
    for gid, meta in clean_groups.items():
        primary_dom = meta["domains"][0]
        all_domains.add(primary_dom)
        by_domain_pool[(primary_dom, meta["public_pool"])].append(gid)

    assigned_split_by_group: dict[str, str] = {}
    per_domain_summary: dict[str, dict[str, int]] = {}

    for dom in sorted(all_domains):
        train_gids = sorted(
            by_domain_pool.get((dom, "train"), []),
            key=lambda g: (_deterministic_group_rank(g, seed), g),
        )
        val_gids = sorted(
            by_domain_pool.get((dom, "validation"), []),
            key=lambda g: (_deterministic_group_rank(g, seed), g),
        )
        avail_tr = len(train_gids)
        avail_va = len(val_gids)

        max_cap = int(max_groups_per_domain) if max_groups_per_domain is not None else 10**9
        eff_total = int(min(float(max_cap), avail_tr / 0.80, avail_va / 0.20))
        q_tune = int(round(0.40 * eff_total))
        q_valid = int(round(0.40 * eff_total))
        q_test = int(round(0.20 * eff_total))
        if q_tune + q_valid > avail_tr:
            half = avail_tr // 2
            q_tune = min(q_tune, half)
            q_valid = min(q_valid, avail_tr - q_tune)
        q_test = min(q_test, avail_va)

        for gid in train_gids[:q_tune]:
            assigned_split_by_group[gid] = "tune"
        for gid in train_gids[q_tune : q_tune + q_valid]:
            assigned_split_by_group[gid] = "valid_confirm"
        for gid in val_gids[:q_test]:
            assigned_split_by_group[gid] = "test"

        per_domain_summary[dom] = {
            "eligible_train_groups": avail_tr,
            "eligible_val_groups": avail_va,
            "tune_groups": q_tune,
            "valid_confirm_groups": q_valid,
            "test_groups": q_test,
        }

    splits: dict[str, list[dict[str, Any]]] = {"tune": [], "valid_confirm": [], "test": []}
    split_groups: dict[str, set[str]] = {"tune": set(), "valid_confirm": set(), "test": set()}
    split_videos: dict[str, set[str]] = {"tune": set(), "valid_confirm": set(), "test": set()}

    for gid, split_name in sorted(assigned_split_by_group.items()):
        meta = clean_groups[gid]
        split_groups[split_name].add(gid)
        for vid in meta["video_ids"]:
            split_videos[split_name].add(vid)
        for rec in meta["records"]:
            rec_copy = dict(rec)
            rec_copy["source_group_id"] = gid
            rec_copy["split"] = split_name
            splits[split_name].append(rec_copy)

    # Strict pairwise zero-overlap assertions
    for a, b in (("tune", "valid_confirm"), ("tune", "test"), ("valid_confirm", "test")):
        overlap_g = split_groups[a] & split_groups[b]
        overlap_v = split_videos[a] & split_videos[b]
        if overlap_g or overlap_v:
            raise AssertionError(
                f"Split leakage between {a} and {b}: groups={sorted(overlap_g)}, videos={sorted(overlap_v)}"
            )

    manifests: dict[str, Any] = {}
    for sname in ("tune", "valid_confirm", "test"):
        gallery_ids = sorted(split_videos[sname])
        src_ids = sorted(split_groups[sname])
        q_rows = sorted(
            splits[sname],
            key=lambda r: (str(r.get("domain") or ""), str(r.get("query_id") or r.get("pair_id") or r.get("video_id") or "")),
        )
        manifests[sname] = {
            "split": sname,
            "seed": int(seed),
            "annotation_version": str(annotation_version),
            "source_groups": src_ids,
            "gallery_video_ids": gallery_ids,
            "records": q_rows,
            "counts": {
                "source_groups": len(src_ids),
                "gallery_videos": len(gallery_ids),
                "records": len(q_rows),
            },
            "checksums": {
                "sources_sha256": _sha256_json(src_ids),
                "gallery_sha256": _sha256_json(gallery_ids),
                "queries_sha256": _sha256_json(q_rows),
            },
        }

    return {
        "seed": int(seed),
        "annotation_version": str(annotation_version),
        "per_domain_summary": per_domain_summary,
        "quarantined_count": len(quarantined),
        "quarantined_groups": quarantined,
        "overlap_verified": True,
        "splits": manifests,
    }


def _load_json_records(path: Path | None, *, default_domain: str, default_kind: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            r = dict(item)
            r.setdefault("domain", default_domain)
            r.setdefault("task_kind", default_kind)
            r.setdefault("query_id", str(r.get("key") or f"{default_domain}_{idx}"))
            out.append(r)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            r = dict(v)
            r.setdefault("video_id", str(k))
            r.setdefault("domain", default_domain)
            r.setdefault("task_kind", default_kind)
            r.setdefault("query_id", str(r.get("key") or k))
            out.append(r)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock 40/40/20 source-disjoint ActivityNet & VideoComp splits.")
    parser.add_argument("--activitynet-captions", type=Path, default=None)
    parser.add_argument("--activitynet-comp", type=Path, default=None)
    parser.add_argument("--youcook2-comp", type=Path, default=None)
    parser.add_argument("--media-hashes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-groups-per-domain", type=int, default=2000)
    parser.add_argument("--annotation-version", type=str, default="v1")
    args = parser.parse_args(list(argv) if argv is not None else None)

    records: list[dict[str, Any]] = []
    records.extend(
        _load_json_records(args.activitynet_captions, default_domain="activitynet", default_kind="captions")
    )
    records.extend(
        _load_json_records(args.activitynet_comp, default_domain="activitynet", default_kind="videocomp")
    )
    records.extend(
        _load_json_records(args.youcook2_comp, default_domain="youcook2", default_kind="videocomp")
    )

    hashes: dict[str, str] | None = None
    if args.media_hashes is not None and args.media_hashes.is_file():
        hashes = json.loads(args.media_hashes.read_text(encoding="utf-8"))

    bundle = assign_source_disjoint_splits(
        records,
        seed=args.seed,
        max_groups_per_domain=args.max_groups_per_domain,
        annotation_version=args.annotation_version,
        file_sha256_by_video_id=hashes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sname, manifest in bundle["splits"].items():
        (args.output_dir / f"{sname}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    summary = {k: v for k, v in bundle.items() if k != "splits"}
    (args.output_dir / "split_lock_receipt.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
