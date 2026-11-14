"""Standalone evaluation script consuming frozen predictions and ground truth."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from videntra.evaluation.evaluator import evaluate_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description="VIDENTRA Offline Evaluator")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground truth JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    def _read_jsonl(p: Path) -> list[dict]:
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    preds = _read_jsonl(args.predictions)
    gts = _read_jsonl(args.ground_truth)

    report = evaluate_predictions(preds, gts)
    report_dict = asdict(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(json.dumps(report_dict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
