from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_V2 = (
    Path(__file__).resolve().parents[1]
    / "bi-encoder-finetuned"
    / "models"
    / "bi_encoder_hnm_v2"
    / "vietnamese-bi-encoder-v2-hnm"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_all_audits(training_base: Path) -> list[dict]:
    preferred_dirs = ["training_data_lsd", "training_data_pldc", "training_data_th"]
    files = [training_base / name / "qa_audit.jsonl" for name in preferred_dirs]
    files = [path for path in files if path.is_file()]
    if not files:
        files = sorted(training_base.glob("**/qa_audit.jsonl"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy qa_audit.jsonl dưới {training_base}")

    rows = []
    for path in files:
        source = path.parent.name
        for row in load_jsonl(path):
            row["_source"] = source
            rows.append(row)
    return rows


def chapter_key(row: dict) -> tuple[str, str]:
    return row.get("_source", ""), row.get("chapter", "Unknown")


def split_by_chapter_3way(
    rows: list[dict], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    by_key = defaultdict(list)
    for row in rows:
        by_key[chapter_key(row)].append(row)

    keys = list(by_key.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_total = len(rows)
    val_target = int(n_total * val_ratio)
    test_target = int(n_total * test_ratio)

    train_rows, val_rows, test_rows = [], [], []
    for key in keys:
        bucket = by_key[key]
        if len(val_rows) < val_target:
            val_rows.extend(bucket)
        elif len(test_rows) < test_target:
            test_rows.extend(bucket)
        else:
            train_rows.extend(bucket)
    return train_rows, val_rows, test_rows


def build_eval_corpus(rows: list[dict]) -> tuple[list[str], list[str], list[int]]:
    corpus_texts = []
    positive_to_idx = {}
    gold_indices = []

    for row in rows:
        pos = row["positive"]
        if pos not in positive_to_idx:
            positive_to_idx[pos] = len(corpus_texts)
            corpus_texts.append(pos)
        gold_indices.append(positive_to_idx[pos])

    queries = [row["query"] for row in rows]
    return queries, corpus_texts, gold_indices


def evaluate_model(
    name: str,
    model_path: str,
    queries: list[str],
    corpus: list[str],
    gold_indices: list[int],
    batch_size: int,
) -> dict:
    model = SentenceTransformer(model_path)
    start = time.perf_counter()
    query_emb = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    corpus_emb = model.encode(
        corpus,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    encode_seconds = time.perf_counter() - start

    query_emb = np.asarray(query_emb, dtype=np.float32)
    corpus_emb = np.asarray(corpus_emb, dtype=np.float32)

    start = time.perf_counter()
    scores = query_emb @ corpus_emb.T
    top_k = min(10, scores.shape[1])
    top_idx_unsorted = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(scores, top_idx_unsorted, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
    search_seconds = time.perf_counter() - start

    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    for row_idx, gold in enumerate(gold_indices):
        ranked = list(top_idx[row_idx])
        if gold in ranked[:1]:
            recall_at_1 += 1
        if gold in ranked[:5]:
            recall_at_5 += 1
        if gold in ranked[:10]:
            recall_at_10 += 1
        if gold in ranked:
            reciprocal_ranks.append(1.0 / (ranked.index(gold) + 1))
        else:
            reciprocal_ranks.append(0.0)

    n = len(queries)
    return {
        "model": name,
        "model_path": model_path,
        "queries": n,
        "corpus": len(corpus),
        "recall@1": recall_at_1 / n,
        "recall@5": recall_at_5 / n,
        "recall@10": recall_at_10 / n,
        "mrr@10": float(np.mean(reciprocal_ranks)),
        "encode_seconds": encode_seconds,
        "search_seconds": search_seconds,
        "seconds_per_query_search": search_seconds / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base Vietnamese Bi-Encoder vs fine-tuned V2 on chapter-disjoint retrieval test set."
    )
    parser.add_argument(
        "--training-base",
        default="data/training_data",
        help="Thư mục chứa training_data_*/qa_audit.jsonl",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "base_vietnamese_bi_encoder=bkai-foundation-models/vietnamese-bi-encoder",
            "bge_m3=BAAI/bge-m3",
            "e5_large=intfloat/multilingual-e5-large",
            f"v2_hard_negative={DEFAULT_MODEL_V2}",
        ],
        help="Danh sách name=model_path. Mặc định so sánh base VN, BGE-M3, E5-Large, V2.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    training_base = Path(args.training_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_audits(training_base)
    train_rows, val_rows, test_rows = split_by_chapter_3way(all_rows, seed=args.seed)
    queries, corpus, gold_indices = build_eval_corpus(test_rows)

    split_info = {
        "total_pairs": len(all_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "test_intents": dict(Counter(row.get("intent_type", "unknown") for row in test_rows)),
        "split_unit": "(source, chapter)",
    }
    print("Split:", split_info)
    print(f"Test queries={len(queries)}, corpus={len(corpus)}")

    model_specs = []
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"Model spec phải có dạng name=path, nhận được: {spec}")
        name, path = spec.split("=", 1)
        model_specs.append((name, path))

    results = []
    for name, path in model_specs:
        print(f"\nEvaluating {name}: {path}")
        result = evaluate_model(name, path, queries, corpus, gold_indices, args.batch_size)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    base = results[0]
    for row in results:
        row["delta_mrr@10_vs_base"] = row["mrr@10"] - base["mrr@10"]
        row["relative_mrr@10_vs_base"] = (
            row["delta_mrr@10_vs_base"] / base["mrr@10"] if base["mrr@10"] else 0.0
        )

    csv_path = out_dir / "compare_biencoder_v2_retrieval.csv"
    json_path = out_dir / "compare_biencoder_v2_retrieval.json"

    fieldnames = [
        "model",
        "queries",
        "corpus",
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr@10",
        "delta_mrr@10_vs_base",
        "relative_mrr@10_vs_base",
        "encode_seconds",
        "search_seconds",
        "seconds_per_query_search",
        "model_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"split": split_info, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
