from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FT_MODEL = (
    REPO_ROOT
    / "bi-encoder-finetuned"
    / "models"
    / "bi_encoder_hnm_v2"
    / "vietnamese-bi-encoder-v2-hnm"
)

SYSTEM_PROMPT = """Bạn là trợ lý học tập cho giáo trình lý luận chính trị Việt Nam.

Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp. Nếu ngữ cảnh không đủ thông tin,
hãy nói rõ rằng không tìm thấy thông tin trong ngữ cảnh. Trả lời ngắn gọn,
đúng trọng tâm, bằng tiếng Việt.

NGỮ CẢNH:
{context}
"""


def clean_text(text: str | None) -> str:
    return " ".join(str(text or "").split())


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
        raise FileNotFoundError(f"Không tìm thấy qa_audit.jsonl canonical dưới {training_base}")

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


def select_eval_rows(rows: list[dict], source: str, sample_size: int, seed: int) -> list[dict]:
    selected = [row for row in rows if row.get("_source") == source]
    if not selected:
        raise ValueError(f"Không có sample test cho source={source}")
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected[: min(sample_size, len(selected))]


def build_corpus(rows: list[dict], source: str) -> list[str]:
    texts = []
    seen = set()
    for row in rows:
        if row.get("_source") != source:
            continue
        text = clean_text(row.get("positive"))
        if text and text not in seen:
            texts.append(text)
            seen.add(text)
    return texts


def retrieve_contexts(
    model_path: str,
    queries: list[str],
    corpus: list[str],
    top_k: int,
    batch_size: int,
) -> tuple[list[list[str]], dict]:
    model = SentenceTransformer(model_path)
    started = time.perf_counter()
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
    encode_seconds = time.perf_counter() - started

    query_emb = np.asarray(query_emb, dtype=np.float32)
    corpus_emb = np.asarray(corpus_emb, dtype=np.float32)
    scores = query_emb @ corpus_emb.T

    k = min(top_k, len(corpus))
    top_idx_unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top_idx_unsorted, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)

    contexts = [[corpus[int(idx)] for idx in row] for row in top_idx]
    stats = {"encode_seconds": encode_seconds, "queries": len(queries), "corpus": len(corpus)}
    return contexts, stats


def generate_answers(
    llm: ChatGoogleGenerativeAI,
    questions: list[str],
    contexts: list[list[str]],
) -> list[str]:
    answers = []
    for idx, (question, ctxs) in enumerate(zip(questions, contexts), 1):
        context = "\n\n".join(f"[Đoạn {i}] {ctx}" for i, ctx in enumerate(ctxs, 1))
        prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nCÂU HỎI:\n{question}"
        print(f"Generating answer {idx}/{len(questions)}")
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        answers.append(clean_text(str(content)))
    return answers


def run_ragas(
    rows: list[dict],
    answers: list[str],
    contexts: list[list[str]],
    llm: ChatGoogleGenerativeAI,
    batch_size: int,
    max_workers: int,
    timeout: int,
) -> tuple[dict, list[dict]]:
    records = []
    for row, answer, ctxs in zip(rows, answers, contexts):
        records.append(
            {
                "question": clean_text(row["query"]),
                "answer": answer,
                "contexts": ctxs,
                "ground_truth": clean_text(row["positive"]),
            }
        )

    dataset = Dataset.from_list(records)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=timeout, max_workers=max_workers, max_retries=2, max_wait=20),
        raise_exceptions=False,
        batch_size=batch_size,
    )
    scores = dict(getattr(result, "_repr_dict", {}))
    if not scores:
        scores = result.to_pandas().select_dtypes(include=["number"]).mean().to_dict()
    return scores, records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Base vs FT retriever with Gemini generation and RAGAS.")
    parser.add_argument("--training-base", default="data/training_data")
    parser.add_argument(
        "--source",
        default="training_data_pldc",
        help="Nguồn đánh giá. Mặc định dùng training_data_pldc vì chapter-disjoint test split seed=42 có đủ mẫu PLDC.",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--llm-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--ragas-batch-size", type=int, default=1)
    parser.add_argument("--ragas-workers", type=int, default=2)
    parser.add_argument("--ragas-timeout", type=int, default=240)
    parser.add_argument("--out-dir", default="artifacts/ragas_retriever_comparison")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "base_vietnamese_bi_encoder=bkai-foundation-models/vietnamese-bi-encoder",
            f"v2_hard_negative={DEFAULT_FT_MODEL}",
        ],
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY chưa có trong environment hoặc .env")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_audits(Path(args.training_base))
    train_rows, val_rows, test_rows = split_by_chapter_3way(all_rows, seed=args.seed)
    eval_rows = select_eval_rows(test_rows, args.source, args.sample_size, args.seed)
    corpus = build_corpus(all_rows, args.source)
    questions = [clean_text(row["query"]) for row in eval_rows]

    split_info = {
        "total_pairs": len(all_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "source": args.source,
        "eval_samples": len(eval_rows),
        "top_k": args.top_k,
        "intent_distribution": dict(Counter(row.get("intent_type", "unknown") for row in eval_rows)),
    }
    print(json.dumps(split_info, ensure_ascii=False, indent=2))

    llm = ChatGoogleGenerativeAI(model=args.llm_model, temperature=0.0, max_tokens=1024)

    summaries = []
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"Model spec phải có dạng name=path, nhận được: {spec}")
        name, model_path = spec.split("=", 1)
        print(f"\n=== {name}: {model_path} ===")

        contexts, retrieval_stats = retrieve_contexts(model_path, questions, corpus, args.top_k, args.batch_size)
        answers = generate_answers(llm, questions, contexts)

        pre_records = []
        for row, answer, ctxs in zip(eval_rows, answers, contexts):
            pre_records.append(
                {
                    "question": clean_text(row["query"]),
                    "answer": answer,
                    "contexts": ctxs,
                    "ground_truth": clean_text(row["positive"]),
                    "model": name,
                }
            )
        write_jsonl(out_dir / f"{name}_generated_records.jsonl", pre_records)

        ragas_scores, records = run_ragas(
            eval_rows,
            answers,
            contexts,
            llm,
            args.ragas_batch_size,
            args.ragas_workers,
            args.ragas_timeout,
        )

        for record, answer, ctxs in zip(records, answers, contexts):
            record["model"] = name
            record["retrieved_contexts"] = ctxs
            record["answer"] = answer

        write_jsonl(out_dir / f"{name}_records.jsonl", records)
        summary = {"model": name, "model_path": model_path, **retrieval_stats, **ragas_scores}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    with open(out_dir / "ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump({"split_info": split_info, "summaries": summaries}, f, ensure_ascii=False, indent=2, default=str)

    fieldnames = sorted({key for row in summaries for key in row.keys()})
    with open(out_dir / "ragas_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
