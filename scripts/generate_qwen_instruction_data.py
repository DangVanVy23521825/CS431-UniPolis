#!/usr/bin/env python3
"""
Generate instruction-tuning data for Qwen LoRA from UniPolis textbooks.

This script does not call any external LLM. It creates controlled synthetic
instruction data from PDF chunks using templates and source-grounded answers.

Output schema is compatible with Notebook/qwen_lora_question_generation.ipynb:
  {
    "id": "...",
    "task": "...",
    "messages": [{"role": "system", ...}, ...],
    "metadata": {...}
  }
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz


SYSTEM_PROMPT = (
    "Bạn là trợ lý học tập tiếng Việt cho các môn lý luận chính trị ở bậc đại học. "
    "Luôn bám sát NGỮ CẢNH được cung cấp, không bịa thông tin. "
    "Chỉ trả về một JSON object hợp lệ, không thêm markdown, không thêm giải thích ngoài JSON. "
    "JSON phải đóng đủ ngoặc và dùng đúng tên trường được yêu cầu."
)

SUBJECTS = [
    {
        "key": "lich_su_dang",
        "subject": "Lịch sử Đảng Cộng sản Việt Nam",
        "pdf": "lich-su-dang-cong-san-viet-nam-giao-trinh.pdf",
    },
    {
        "key": "phap_luat_dai_cuong",
        "subject": "Pháp luật đại cương",
        "pdf": "phap-luat-dai-cuong-giao-trinh.pdf",
    },
    {
        "key": "triet_hoc_mac_lenin",
        "subject": "Triết học Mác-Lênin",
        "pdf": "triet-hoc-mac-lenin-giao-trinh.pdf",
    },
    {
        "key": "kinh_te_chinh_tri",
        "subject": "Kinh tế Chính trị Mác-Lênin",
        "pdf": "kinh-te-chinh-tri-mac-lenin-giao-trinh.pdf",
    },
    {
        "key": "chu_nghia_xa_hoi_khoa_hoc",
        "subject": "Chủ nghĩa Xã hội Khoa học",
        "pdf": "chu-nghia-xa-hoi-khoa-hoc-giao-trinh.pdf",
    },
    {
        "key": "tu_tuong_hcm",
        "subject": "Tư tưởng Hồ Chí Minh",
        "pdf": "tu-tuong-hcm-giao-trinh.pdf",
    },
]

TASK_PLAN = [
    ("grounded_qa", 75),
    ("mcq_generation", 60),
    ("question_generation", 45),
    ("summary", 40),
    ("flashcard", 30),
]

TASK_PLAN_NO_FLASHCARD = [
    ("grounded_qa", 90),
    ("mcq_generation", 75),
    ("question_generation", 50),
    ("summary", 35),
]

QUESTION_STEMS = {
    "lich_su_dang": [
        "Trình bày nội dung chính được nêu trong đoạn giáo trình về lịch sử Đảng.",
        "Đoạn giáo trình làm rõ vấn đề gì trong quá trình lãnh đạo của Đảng?",
        "Nêu ý nghĩa hoặc nội dung trọng tâm của đoạn giáo trình sau.",
    ],
    "phap_luat_dai_cuong": [
        "Trình bày khái niệm hoặc nội dung pháp lý chính trong đoạn giáo trình.",
        "Đoạn giáo trình làm rõ quy định hoặc đặc điểm pháp luật nào?",
        "Nêu nội dung trọng tâm về pháp luật được trình bày trong đoạn sau.",
    ],
    "triet_hoc_mac_lenin": [
        "Trình bày luận điểm triết học chính được nêu trong đoạn giáo trình.",
        "Đoạn giáo trình làm rõ khái niệm, phạm trù hoặc quy luật triết học nào?",
        "Nêu nội dung trọng tâm theo quan điểm triết học Mác-Lênin trong đoạn sau.",
    ],
    "kinh_te_chinh_tri": [
        "Trình bày nội dung kinh tế chính trị chính được nêu trong đoạn giáo trình.",
        "Đoạn giáo trình làm rõ khái niệm hoặc quy luật kinh tế chính trị nào?",
        "Nêu luận điểm trọng tâm về kinh tế chính trị Mác-Lênin trong đoạn sau.",
    ],
    "chu_nghia_xa_hoi_khoa_hoc": [
        "Trình bày nội dung lý luận chính được nêu trong đoạn giáo trình.",
        "Đoạn giáo trình làm rõ vấn đề nào của chủ nghĩa xã hội khoa học?",
        "Nêu luận điểm trọng tâm về chủ nghĩa xã hội khoa học trong đoạn sau.",
    ],
    "tu_tuong_hcm": [
        "Trình bày nội dung chính trong tư tưởng Hồ Chí Minh được nêu ở đoạn sau.",
        "Đoạn giáo trình làm rõ quan điểm nào của Hồ Chí Minh?",
        "Nêu luận điểm trọng tâm trong tư tưởng Hồ Chí Minh từ đoạn giáo trình sau.",
    ],
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    subject_key: str
    subject: str
    source_pdf: str
    page_start: int
    page_end: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"Downloaded by [^\n]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"lOMoARcPSD\|[^\n]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Scan to open on Studocu", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Studocu is not sponsored or endorsed[^\n]*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,4}\b\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for part in parts:
        part = clean_text(part)
        if word_count(part) >= 6:
            sentences.append(part)
    return sentences


def first_sentences(text: str, n: int = 2, max_chars: int = 850) -> str:
    sentences = split_sentences(text)
    selected = " ".join(sentences[:n]) if sentences else clean_text(text)
    if len(selected) > max_chars:
        selected = selected[:max_chars].rsplit(" ", 1)[0] + "..."
    return selected.strip()


def starts_mid_sentence(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return True
    first = text[0]
    if first.isdigit() or first in "([{\"'“”‘’":
        return False
    return first.islower()


def has_repetitive_ngram(text: str, n: int = 4, threshold: int = 3) -> bool:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < n * threshold:
        return False
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(words[i : i + n]) for i in range(len(words) - n + 1)
    )
    return bool(counts and max(counts.values()) >= threshold)


def has_bad_pdf_noise(text: str) -> bool:
    text_l = text.lower()
    bad_markers = [
        "downloaded by",
        "st udocu",
        "studocu",
        "lomoarcpsd",
        "scan to open",
        "vui lòng cung cấp",
    ]
    if any(m in text_l for m in bad_markers):
        return True
    if text.count("...") >= 4:
        return True
    if len(re.findall(r"\.{2,}|_{2,}|-{4,}", text)) >= 4:
        return True
    return False


def is_good_chunk(text: str, min_words: int = 90, max_words: int = 240) -> bool:
    text = clean_text(text)
    n_words = word_count(text)
    if n_words < min_words or n_words > max_words:
        return False
    if starts_mid_sentence(text):
        return False
    if has_bad_pdf_noise(text):
        return False
    if has_repetitive_ngram(text):
        return False
    return True


def context_excerpt(text: str, max_chars: int = 1800) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind(";"))
    if last > max_chars * 0.55:
        cut = cut[: last + 1]
    return cut.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for idx in range(len(doc)):
        text = clean_text(doc[idx].get_text())
        if word_count(text) >= 60:
            pages.append((idx + 1, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    subject_key: str,
    subject: str,
    source_pdf: str,
    min_words: int = 90,
    max_words: int = 220,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_idx = 1
    for page_no, text in pages:
        sentences = split_sentences(text)
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            n_words = word_count(sentence)
            if current and current_words + n_words > max_words:
                chunk_text = " ".join(current).strip()
                if is_good_chunk(chunk_text, min_words=min_words, max_words=max_words):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{subject_key}_{chunk_idx:04d}",
                            subject_key=subject_key,
                            subject=subject,
                            source_pdf=source_pdf,
                            page_start=page_no,
                            page_end=page_no,
                            text=chunk_text,
                        )
                    )
                    chunk_idx += 1
                current = []
                current_words = 0

            current.append(sentence)
            current_words += n_words

        if current:
            chunk_text = " ".join(current).strip()
            if is_good_chunk(chunk_text, min_words=min_words, max_words=max_words):
                chunks.append(
                    Chunk(
                        chunk_id=f"{subject_key}_{chunk_idx:04d}",
                        subject_key=subject_key,
                        subject=subject,
                        source_pdf=source_pdf,
                        page_start=page_no,
                        page_end=page_no,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1
    return chunks


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def record_id(chunk: Chunk, task: str, n: int) -> str:
    return f"{chunk.chunk_id}_{task}_{n:03d}"


def base_metadata(chunk: Chunk) -> dict:
    return {
        "subject": chunk.subject,
        "subject_key": chunk.subject_key,
        "source_pdf": chunk.source_pdf,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_id": chunk.chunk_id,
        "generator": "template_v1_pdf_grounded",
    }


def make_messages(user: str, assistant_obj: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json_dumps(assistant_obj)},
    ]


def make_grounded_qa(chunk: Chunk, n: int, rng: random.Random) -> dict:
    question = rng.choice(QUESTION_STEMS[chunk.subject_key])
    context = context_excerpt(chunk.text)
    answer = first_sentences(chunk.text, n=3)
    user = f"""Nhiệm vụ: Trả lời câu hỏi dựa HOÀN TOÀN vào NGỮ CẢNH cho môn {chunk.subject}.
Nếu ngữ cảnh không đủ thông tin, hãy nói rõ là không đủ thông tin.
Chỉ trả về JSON hợp lệ, không markdown, không text ngoài JSON. JSON phải có đủ các trường: task, answer, evidence, source_page.

CÂU HỎI:
{question}

NGỮ CẢNH:
{context}"""
    assistant = {
        "task": "grounded_qa",
        "answer": answer,
        "evidence": first_sentences(chunk.text, n=1),
        "source_page": str(chunk.page_start),
    }
    return {
        "id": record_id(chunk, "grounded_qa", n),
        "task": "grounded_qa",
        "messages": make_messages(user, assistant),
        "target_question": question,
        "metadata": base_metadata(chunk),
    }


def make_question_generation(chunk: Chunk, n: int, rng: random.Random) -> dict:
    question = rng.choice(QUESTION_STEMS[chunk.subject_key])
    if rng.random() < 0.35:
        question = "Nêu nội dung trọng tâm của đoạn giáo trình sau và giải thích ngắn gọn."
    context = context_excerpt(chunk.text)
    user = f"""Nhiệm vụ: Dựa vào NGỮ CẢNH, hãy tạo 1 câu hỏi ôn tập cho môn {chunk.subject}.
Câu hỏi phải bám sát nội dung giáo trình, không dùng cụm 'theo đoạn trên'.
Chỉ trả về JSON hợp lệ, không markdown, không text ngoài JSON. JSON phải có đủ các trường: task, subject, question_type, question, expected_evidence.

NGỮ CẢNH:
{context}"""
    assistant = {
        "task": "question_generation",
        "subject": chunk.subject,
        "question_type": "short_answer",
        "question": question,
        "expected_evidence": first_sentences(chunk.text, n=1),
    }
    return {
        "id": record_id(chunk, "question_generation", n),
        "task": "question_generation",
        "messages": make_messages(user, assistant),
        "target_question": question,
        "metadata": base_metadata(chunk),
    }


def make_summary(chunk: Chunk, n: int) -> dict:
    context = context_excerpt(chunk.text)
    summary = first_sentences(chunk.text, n=3)
    user = f"""Nhiệm vụ: Tóm tắt NGỮ CẢNH sau thành 2-3 ý chính để hỗ trợ ôn tập môn {chunk.subject}.
Chỉ dùng thông tin trong ngữ cảnh.
Chỉ trả về JSON hợp lệ, không markdown, không text ngoài JSON. JSON phải có đủ các trường: task, summary, key_points, source_page.

NGỮ CẢNH:
{context}"""
    key_points = split_sentences(summary)[:3]
    assistant = {
        "task": "summary",
        "summary": summary,
        "key_points": key_points,
        "source_page": str(chunk.page_start),
    }
    return {
        "id": record_id(chunk, "summary", n),
        "task": "summary",
        "messages": make_messages(user, assistant),
        "metadata": base_metadata(chunk),
    }


def make_flashcard(chunk: Chunk, n: int, rng: random.Random) -> dict:
    front = rng.choice(QUESTION_STEMS[chunk.subject_key])
    back = first_sentences(chunk.text, n=2)
    context = context_excerpt(chunk.text)
    user = f"""Nhiệm vụ: Tạo 1 flashcard ôn tập từ NGỮ CẢNH cho môn {chunk.subject}.
Mặt trước là câu hỏi ngắn, mặt sau là câu trả lời bám sát ngữ cảnh.
Chỉ trả về JSON hợp lệ, không markdown, không text ngoài JSON. JSON phải có đủ các trường: task, front, back, source_page.

NGỮ CẢNH:
{context}"""
    assistant = {
        "task": "flashcard",
        "front": front,
        "back": back,
        "source_page": str(chunk.page_start),
    }
    return {
        "id": record_id(chunk, "flashcard", n),
        "task": "flashcard",
        "messages": make_messages(user, assistant),
        "metadata": base_metadata(chunk),
    }


def make_mcq(chunk: Chunk, n: int, distractor_pool: list[Chunk], rng: random.Random) -> dict | None:
    distractors = [c for c in distractor_pool if c.chunk_id != chunk.chunk_id]
    if len(distractors) < 3:
        return None
    selected = rng.sample(distractors, 3)
    correct = first_sentences(chunk.text, n=1, max_chars=170)
    wrongs = [first_sentences(c.text, n=1, max_chars=170) for c in selected]
    if any(len(opt) > 190 for opt in [correct, *wrongs]):
        return None
    if any(has_repetitive_ngram(opt, n=3, threshold=2) for opt in [correct, *wrongs]):
        return None
    if len({correct, *wrongs}) < 4:
        return None

    options_raw = [("correct", correct), *[("wrong", w) for w in wrongs]]
    rng.shuffle(options_raw)
    labels = ["A", "B", "C", "D"]
    options = {label: value for label, (_, value) in zip(labels, options_raw)}
    answer = next(label for label, (kind, _) in zip(labels, options_raw) if kind == "correct")

    context = context_excerpt(chunk.text)
    user = f"""Nhiệm vụ: Dựa vào NGỮ CẢNH, hãy tạo 1 câu hỏi trắc nghiệm 4 lựa chọn cho môn {chunk.subject}.
Câu hỏi phải kiểm tra khả năng nhận biết luận điểm đúng trong ngữ cảnh. Mỗi lựa chọn phải ngắn gọn, không lặp cụm từ dài.
Chỉ trả về JSON hợp lệ, không markdown, không text ngoài JSON. JSON phải có đủ các trường: task, question, options, answer, explanation, source_page.

NGỮ CẢNH:
{context}"""
    assistant = {
        "task": "mcq_generation",
        "question": "Nhận định nào sau đây phản ánh đúng nội dung của đoạn giáo trình?",
        "options": options,
        "answer": answer,
        "explanation": correct,
        "source_page": str(chunk.page_start),
    }
    return {
        "id": record_id(chunk, "mcq_generation", n),
        "task": "mcq_generation",
        "messages": make_messages(user, assistant),
        "metadata": base_metadata(chunk),
    }


def cycle_chunks(chunks: list[Chunk], total: int, rng: random.Random) -> Iterable[Chunk]:
    if not chunks:
        return
    shuffled = chunks[:]
    rng.shuffle(shuffled)
    for i in range(total):
        yield shuffled[i % len(shuffled)]


def build_records_for_subject(
    chunks: list[Chunk], per_subject: int, rng: random.Random, task_plan: list[tuple[str, int]]
) -> list[dict]:
    records: list[dict] = []
    by_subject = chunks[:]
    task_counts = dict(task_plan)
    planned = sum(task_counts.values())
    if planned != per_subject:
        scale = per_subject / planned
        task_counts = {k: max(1, round(v * scale)) for k, v in task_counts.items()}
        diff = per_subject - sum(task_counts.values())
        task_counts["grounded_qa"] += diff

    n_by_task: Counter[str] = Counter()
    for task_name, count in task_counts.items():
        for chunk in cycle_chunks(by_subject, count, rng):
            n_by_task[task_name] += 1
            n = n_by_task[task_name]
            if task_name == "grounded_qa":
                records.append(make_grounded_qa(chunk, n, rng))
            elif task_name == "question_generation":
                records.append(make_question_generation(chunk, n, rng))
            elif task_name == "summary":
                records.append(make_summary(chunk, n))
            elif task_name == "flashcard":
                records.append(make_flashcard(chunk, n, rng))
            elif task_name == "mcq_generation":
                rec = make_mcq(chunk, n, by_subject, rng)
                if rec is not None:
                    records.append(rec)

    while len(records) < per_subject:
        chunk = rng.choice(by_subject)
        n_by_task["grounded_qa"] += 1
        records.append(make_grounded_qa(chunk, n_by_task["grounded_qa"], rng))
    return records[:per_subject]


def split_records(records: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_subject[rec["metadata"]["subject_key"]].append(rec)

    train, val, test = [], [], []
    for subject_records in by_subject.values():
        rng.shuffle(subject_records)
        n = len(subject_records)
        n_val = round(n * 0.1)
        n_test = round(n * 0.1)
        val.extend(subject_records[:n_val])
        test.extend(subject_records[n_val : n_val + n_test])
        train.extend(subject_records[n_val + n_test :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty_sample(path: Path, rows: list[dict], n: int = 12) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows[:n], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen SFT instruction data from UniPolis PDFs"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing PDFs")
    parser.add_argument(
        "--output-dir",
        default="data/qwen_lora_instruction_data",
        help="Output directory for JSONL files",
    )
    parser.add_argument("--total", type=int, default=1500, help="Total samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-flashcard",
        action="store_true",
        help="Do not generate flashcard task; redistribute samples to other tasks.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    per_subject = args.total // len(SUBJECTS)
    remainder = args.total % len(SUBJECTS)
    task_plan = TASK_PLAN_NO_FLASHCARD if args.no_flashcard else TASK_PLAN

    all_records: list[dict] = []
    all_chunks: list[Chunk] = []
    chunk_stats = {}

    for idx, cfg in enumerate(SUBJECTS):
        pdf_path = data_dir / cfg["pdf"]
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(
            pages=pages,
            subject_key=cfg["key"],
            subject=cfg["subject"],
            source_pdf=cfg["pdf"],
        )
        if not chunks:
            raise RuntimeError(f"No chunks generated for {pdf_path}")
        all_chunks.extend(chunks)
        target = per_subject + (1 if idx < remainder else 0)
        records = build_records_for_subject(chunks, target, rng, task_plan)
        all_records.extend(records)
        chunk_stats[cfg["key"]] = {
            "subject": cfg["subject"],
            "pdf": cfg["pdf"],
            "pages": len(pages),
            "chunks": len(chunks),
            "records": len(records),
        }
        print(
            f"{cfg['subject']}: pages={len(pages)}, chunks={len(chunks)}, records={len(records)}"
        )

    train, val, test = split_records(all_records, args.seed)
    write_jsonl(out_dir / "train_sft_messages.jsonl", train)
    write_jsonl(out_dir / "val_sft_messages.jsonl", val)
    write_jsonl(out_dir / "test_sft_messages.jsonl", test)
    write_jsonl(out_dir / "all_sft_messages.jsonl", all_records)
    write_pretty_sample(out_dir / "sample_preview.json", all_records)

    manifest = {
        "total_records": len(all_records),
        "train_records": len(train),
        "val_records": len(val),
        "test_records": len(test),
        "task_distribution": Counter(r["task"] for r in all_records),
        "subject_distribution": Counter(r["metadata"]["subject_key"] for r in all_records),
        "chunk_stats": chunk_stats,
        "schema": "chat_messages_jsonl_v1",
        "task_plan": task_plan,
        "no_flashcard": args.no_flashcard,
        "system_prompt": SYSTEM_PROMPT,
        "note": (
            "Synthetic instruction data generated from textbook chunks using "
            "source-grounded templates. Recommended: manually review a subset "
            "or distill additional samples with a teacher LLM for higher quality."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nOutput:")
    print(f"  {out_dir / 'train_sft_messages.jsonl'}")
    print(f"  {out_dir / 'val_sft_messages.jsonl'}")
    print(f"  {out_dir / 'test_sft_messages.jsonl'}")
    print(f"  {out_dir / 'manifest.json'}")
    print("\nSummary:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
