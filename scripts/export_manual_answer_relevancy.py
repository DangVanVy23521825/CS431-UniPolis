#!/usr/bin/env python3
"""Export generated_records.jsonl → CSV để chấm Answer Relevancy thủ công."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "ragas_retriever_comparison_pldc_30_retry"

MODEL_FILES = [
    ("base_vietnamese_bi_encoder", "base_vietnamese_bi_encoder_generated_records.jsonl"),
    ("v2_hard_negative", "v2_hard_negative_generated_records.jsonl"),
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preview(text: str, max_len: int = 400) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def export_model_csv(rows: list[dict], model: str, out_path: Path) -> None:
    fieldnames = [
        "model",
        "idx",
        "question",
        "answer",
        "contexts_preview",
        "ground_truth_preview",
        "manual_relevancy",
        "notes",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            contexts = row.get("contexts") or []
            ctx_preview = " || ".join(preview(c, 180) for c in contexts[:3])
            if len(contexts) > 3:
                ctx_preview += f" || ... (+{len(contexts) - 3} đoạn)"
            writer.writerow(
                {
                    "model": model,
                    "idx": idx,
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "contexts_preview": ctx_preview,
                    "ground_truth_preview": preview(row.get("ground_truth", ""), 400),
                    "manual_relevancy": "",
                    "notes": "",
                }
            )


def export_combined(rows_by_model: list[tuple[str, list[dict]]], out_path: Path) -> None:
    fieldnames = [
        "model",
        "idx",
        "question",
        "answer",
        "contexts_preview",
        "ground_truth_preview",
        "manual_relevancy",
        "notes",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, rows in rows_by_model:
            for idx, row in enumerate(rows, 1):
                contexts = row.get("contexts") or []
                ctx_preview = " || ".join(preview(c, 180) for c in contexts[:3])
                if len(contexts) > 3:
                    ctx_preview += f" || ... (+{len(contexts) - 3} đoạn)"
                writer.writerow(
                    {
                        "model": model,
                        "idx": idx,
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "contexts_preview": ctx_preview,
                        "ground_truth_preview": preview(row.get("ground_truth", ""), 400),
                        "manual_relevancy": "",
                        "notes": "",
                    }
                )


def write_readme(out_dir: Path) -> None:
    text = """# Chấm Answer Relevancy thủ công

## File
- `manual_answer_relevancy_all.csv` — gộp base + V2
- `manual_answer_relevancy_<model>.csv` — từng model

## Cách chấm (cột `manual_relevancy`)
Chỉ cần đọc `question` + `answer` (cột ground_truth/contexts chỉ tham khảo).

| Điểm | Ý nghĩa |
|------|---------|
| 1.0 | Trả lời đúng trọng tâm, không lan man |
| 0.5 | Có liên quan nhưng thiếu/sai trọng tâm hoặc thừa ý |
| 0.0 | Lạc đề hoặc không trả lời câu hỏi |

## Tính trung bình
Sau khi điền xong, chạy:

```bash
python scripts/summarize_manual_answer_relevancy.py --csv artifacts/.../manual_answer_relevancy_all.csv
```
"""
    (out_dir / "MANUAL_SCORING_README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSV chấm Answer Relevancy thủ công.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="Mặc định = artifact-dir")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    out_dir = args.out_dir or artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model: list[tuple[str, list[dict]]] = []
    for model, filename in MODEL_FILES:
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu file: {path}")
        rows = load_jsonl(path)
        per_model_csv = out_dir / f"manual_answer_relevancy_{model}.csv"
        export_model_csv(rows, model, per_model_csv)
        rows_by_model.append((model, rows))
        print(f"Wrote {per_model_csv} ({len(rows)} rows)")

    combined = out_dir / "manual_answer_relevancy_all.csv"
    export_combined(rows_by_model, combined)
    write_readme(out_dir)
    print(f"Wrote {combined}")
    print(f"Wrote {out_dir / 'MANUAL_SCORING_README.md'}")


if __name__ == "__main__":
    main()
