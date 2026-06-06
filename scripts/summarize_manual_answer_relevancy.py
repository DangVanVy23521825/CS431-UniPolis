#!/usr/bin/env python3
"""Tổng hợp điểm Answer Relevancy thủ công từ CSV đã điền."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    REPO_ROOT
    / "artifacts"
    / "ragas_retriever_comparison_pldc_30_retry"
    / "manual_answer_relevancy_all.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp manual_relevancy theo model.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    by_model: dict[str, list[float]] = {}
    missing: list[tuple[str, int]] = []

    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            raw = (row.get("manual_relevancy") or "").strip()
            if not raw:
                missing.append((model, int(row["idx"])))
                continue
            by_model.setdefault(model, []).append(float(raw))

    if missing:
        print(f"WARNING: thiếu điểm ở {len(missing)} dòng (vd {missing[:3]}...)")

    summaries = []
    print("\nAnswer Relevancy (manual)")
    print("-" * 50)
    for model, scores in sorted(by_model.items()):
        avg = sum(scores) / len(scores) if scores else float("nan")
        summaries.append({"model": model, "n_scored": len(scores), "answer_relevancy_manual": avg})
        print(f"  {model}: {avg:.4f}  (n={len(scores)})")

    out_json = args.out_json or args.csv.with_name("manual_answer_relevancy_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "missing_rows": missing}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
