"""Evaluator-side readiness audit; does not run inference or move media."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

if __package__:
    from .audit_qvh_coverage import audit as coverage_audit
else:
    from audit_qvh_coverage import audit as coverage_audit


def audit(root: Path) -> dict:
    qvh = root / "data/raw/qvhighlights/quarantine_source_disjoint_20261110/source_receipts.jsonl"
    records = [json.loads(line) for line in qvh.read_text(encoding="utf-8").splitlines() if line.strip()]
    coverage = coverage_audit(root / "work/evaluation-only/splits/qvh_source_disjoint_20261110.json", "test",
        root / "test_queries/qvhighlights/highlight_val_release.jsonl", root / "data/processed/metadata.db",
        root / "data/processed/faiss_index.bin")
    with sqlite3.connect(f"{(root / 'data/processed/metadata.db').resolve().as_uri()}?mode=ro", uri=True) as conn:
        active = conn.execute("SELECT weights_hash, checksum FROM IndexManifest WHERE is_active=1").fetchall()
    proposal = json.loads((root / "work/audit/timeline_repair_proposal_20261110.json").read_text(encoding="utf-8"))
    return {"status": "NOT_READY", "qvh_receipt_count": len(records),
            "rights_status_counts": {status: sum(row.get("rights_status") == status for row in records)
                                     for status in sorted({row.get("rights_status", "missing") for row in records})},
            "test_coverage": coverage["coverage"]["qvh_test"],
            "active_index_manifest_count": len(active),
            "active_index_has_weights_and_index_hash": len(active) == 1 and all(bool(x) for x in active[0]),
            "timeline_proposal_rows_review_required": len(proposal["proposed_rows"]),
            "full_a5_model_and_runtime_provenance": "UNVERIFIED",
            "relation_action_role_verification": "UNVERIFIED",
            "limitations": ["Audit is evaluator-only and does not authorize asset use or choose retrieval candidates.",
                            "The existing snapshot and review-only timeline proposal are not modified."]}


if __name__ == "__main__":
    print(json.dumps(audit(Path.cwd()), indent=2, ensure_ascii=False))
