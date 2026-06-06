"""
exam_generator.py — Module tạo đề thi từ RAG pipeline.

Tích hợp với HistoryRAG hiện có, không thay đổi rag_engine.py.
Dùng single-prompt approach (MVP + V1 compatible).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ExamQuestion:
    index: int
    q_type: str          # mcq | true_false | fill_blank | short_answer | scenario
    question: str
    options: Optional[dict] = None       # MCQ only: {"A": ..., "B": ..., ...}
    answer: Any = None                   # str hoặc bool
    rubric: Optional[str] = None         # short_answer / scenario gợi ý chấm
    source_page: Optional[str] = None
    chunk_uid: Optional[str] = None
    quality_score: float = 0.0
    difficulty: str = "medium"           # easy | medium | hard


@dataclass
class ExamResult:
    questions: list[ExamQuestion] = field(default_factory=list)
    refused: bool = False
    refuse_reason: str = ""
    elapsed: float = 0.0
    subject: str = ""
    topic: str = ""
    n_sources: int = 0
    avg_quality: float = 0.0


# ── Prompt templates ───────────────────────────────────────────────────────────

# BUG 1 FIX: Tách context ra khỏi system prompt thành human prompt riêng.
# LangChain dùng str.format() nên mọi { } trong system prompt phải được escape
# thành {{ }}. Nhưng {context} chứa văn bản tự do có thể có { } → crash.
# Giải pháp: system prompt chỉ chứa các biến tĩnh (subject, n_questions,
# difficulty, type_distribution). Context đưa vào human prompt.

_EXAM_SYSTEM_PROMPT = """Bạn là chuyên gia ra đề thi đại học cho môn {subject}.
Nhiệm vụ: Tạo {n_questions} câu hỏi chất lượng cao dựa HOÀN TOÀN vào NGỮ CẢNH ở cuối tin nhắn.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng thông tin có trong NGỮ CẢNH. Không được bịa đặt.
2. Nếu NGỮ CẢNH có ít hơn 3 đoạn liên quan, trả về JSON: {{"refused": true, "reason": "Mô tả lý do"}}
3. Mỗi câu hỏi PHẢI có trường source_page lấy từ metadata [Đoạn X - Trang Y] trong NGỮ CẢNH.
4. quality_score từ 0.0 đến 1.0: 0.9+ rất tốt, 0.7-0.9 tốt, dưới 0.7 mơ hồ.
5. Trả về JSON hợp lệ THUẦN TÚY, KHÔNG có markdown fence, KHÔNG có text ngoài JSON.
6. Độ khó mong muốn: {difficulty}
7. Phân bổ loại câu hỏi: {type_distribution}

OUTPUT JSON SCHEMA (giữ đúng tên field):
{{"refused": false, "questions": [{{"index": 1, "q_type": "mcq|true_false|fill_blank|short_answer|scenario", "question": "...", "options": {{"A":"...","B":"...","C":"...","D":"..."}}, "answer": "B", "rubric": null, "difficulty": "easy|medium|hard", "source_page": "12", "quality_score": 0.87}}]}}

Với true_false: answer là true hoặc false (boolean JSON, không phải string).
Với fill_blank/short_answer/scenario: bỏ trường options, dùng rubric để gợi ý chấm."""

# BUG 1 FIX (tiếp): context đưa vào human prompt, an toàn với mọi ký tự đặc biệt
_EXAM_HUMAN_PROMPT = """Hãy tạo {n_questions} câu hỏi{topic_clause} cho môn {subject}.
Đảm bảo mỗi câu có source_page và quality_score.

NGỮ CẢNH:
{context}"""


# ── Type distribution helper ───────────────────────────────────────────────────

def _build_type_distribution(ratios: dict[str, int], n_total: int) -> tuple[str, dict]:
    """
    Từ ratios (e.g. {"mcq": 50, "true_false": 20, ...}) và n_total,
    trả về (description_str, counts_dict).
    BUG FIX: tổng count luôn = n_total, không bị lệch do rounding.
    """
    total_pct = sum(ratios.values()) or 100
    counts: dict[str, int] = {}
    labels = {
        "mcq": "Trắc nghiệm (MCQ)",
        "true_false": "Đúng/Sai",
        "fill_blank": "Điền khuyết",
        "short_answer": "Tự luận ngắn",
        "scenario": "Câu hỏi tình huống",
    }
    items = [(k, v) for k, v in ratios.items() if v > 0]
    allocated = 0
    for i, (qtype, pct) in enumerate(items):
        if i == len(items) - 1:
            n = max(0, n_total - allocated)
        else:
            n = max(0, round(n_total * pct / total_pct))
        counts[qtype] = n
        allocated += n
    # Loại có pct=0 không đưa vào
    parts = [f"{labels.get(t, t)}: {c} câu" for t, c in counts.items() if c > 0]
    desc = " | ".join(parts) if parts else f"Tổng hợp: {n_total} câu"
    return desc, counts


# ── Main ExamGenerator ─────────────────────────────────────────────────────────

class ExamGenerator:
    """
    Bọc quanh HistoryRAG để sinh đề thi có nguồn trích dẫn.
    Sử dụng single-prompt approach — không thay đổi rag_engine.py.
    """

    def __init__(
        self,
        rag_engine,                           # HistoryRAG instance
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.4,             # Cao hơn chat để đa dạng câu hỏi
        max_tokens: int = 3000,
    ) -> None:
        self.rag = rag_engine
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        subject: str,
        n_questions: int = 8,
        topic: str = "",
        difficulty: str = "medium",          # easy | medium | hard | mixed
        type_ratios: Optional[dict[str, int]] = None,  # {"mcq": 50, "true_false": 20, ...}
    ) -> ExamResult:
        """
        Sinh đề thi.
        Trả về ExamResult kể cả khi từ chối (refused=True).
        """
        if type_ratios is None:
            type_ratios = {"mcq": 50, "true_false": 20, "fill_blank": 15, "short_answer": 15}

        t0 = time.time()

        # 1. Retrieve context
        query = f"{subject} {topic}".strip() if topic else subject
        docs: list[Document] = self.rag._hybrid_retrieve_then_rerank(query)

        # 2. Build inputs
        context_str = self.rag._format_context(docs)
        type_desc, _ = _build_type_distribution(type_ratios, n_questions)
        topic_clause = f" về chủ đề '{topic}'" if topic else ""

        # 3. Build chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", _EXAM_SYSTEM_PROMPT),
            ("human", _EXAM_HUMAN_PROMPT),
        ])
        chain = prompt | self.llm | StrOutputParser()

        raw = chain.invoke({
            "subject": subject,
            "n_questions": n_questions,
            "context": context_str,
            "difficulty": difficulty,
            "type_distribution": type_desc,
            "topic_clause": topic_clause,
        })

        elapsed = time.time() - t0

        # 4. Parse & validate
        return self._parse_result(
            raw=raw,
            elapsed=elapsed,
            subject=subject,
            topic=topic,
            n_sources=len(docs),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_result(
        raw: str,
        elapsed: float,
        subject: str,
        topic: str,
        n_sources: int,
    ) -> ExamResult:
        """Parse JSON output từ LLM, xử lý các lỗi format."""
        # Strip markdown fences nếu model quên
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: tìm JSON object đầu tiên trong string
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return ExamResult(
                        refused=True,
                        refuse_reason="LLM trả về định dạng không hợp lệ. Hãy thử lại.",
                        elapsed=elapsed,
                        subject=subject,
                        topic=topic,
                        n_sources=n_sources,
                    )
            else:
                return ExamResult(
                    refused=True,
                    refuse_reason="Không parse được JSON từ LLM.",
                    elapsed=elapsed,
                    subject=subject,
                    topic=topic,
                    n_sources=n_sources,
                )

        if data.get("refused"):
            return ExamResult(
                refused=True,
                refuse_reason=data.get("reason", "Không đủ ngữ cảnh"),
                elapsed=elapsed,
                subject=subject,
                topic=topic,
                n_sources=n_sources,
            )

        questions: list[ExamQuestion] = []
        for item in data.get("questions", []):
            try:
                q = ExamQuestion(
                    index=item.get("index", len(questions) + 1),
                    q_type=item.get("q_type", "mcq"),
                    question=item.get("question", ""),
                    options=item.get("options"),
                    answer=item.get("answer"),
                    rubric=item.get("rubric"),
                    source_page=str(item.get("source_page", "?")),
                    quality_score=float(item.get("quality_score", 0.5)),
                    difficulty=item.get("difficulty", "medium"),
                )
                questions.append(q)
            except Exception:
                continue  # Skip malformed question

        avg_quality = (
            sum(q.quality_score for q in questions) / len(questions)
            if questions else 0.0
        )

        return ExamResult(
            questions=questions,
            refused=False,
            elapsed=elapsed,
            subject=subject,
            topic=topic,
            n_sources=n_sources,
            avg_quality=round(avg_quality, 2),
        )


# ── Export helpers ─────────────────────────────────────────────────────────────

TYPE_LABELS = {
    "mcq": "Trắc nghiệm",
    "true_false": "Đúng/Sai",
    "fill_blank": "Điền khuyết",
    "short_answer": "Tự luận ngắn",
    "scenario": "Tình huống",
}

DIFF_LABELS = {
    "easy": "Dễ",
    "medium": "Vừa",
    "hard": "Khó",
}


def export_exam_markdown(result: ExamResult) -> str:
    """Export đề thi thành Markdown đẹp."""
    from datetime import datetime
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = [
        f"# Đề thi — {result.subject}",
        "",
        f"- **Chủ đề:** {result.topic or 'Tổng hợp'}",
        f"- **Số câu:** {len(result.questions)}",
        f"- **Thời gian tạo:** {now}",
        f"- **Avg quality score:** {result.avg_quality:.2f}",
        "",
        "---",
        "",
        "## Phần câu hỏi",
        "",
    ]

    for q in result.questions:
        type_label = TYPE_LABELS.get(q.q_type, q.q_type)
        diff_label = DIFF_LABELS.get(q.difficulty, q.difficulty)
        lines.append(f"**Câu {q.index}** _{type_label} · {diff_label} · Trang {q.source_page} · Score: {q.quality_score:.2f}_")
        lines.append("")
        lines.append(q.question)
        lines.append("")

        if q.q_type == "mcq" and q.options:
            for letter, text in q.options.items():
                lines.append(f"- {letter}. {text}")
            lines.append("")

        lines.append("")

    lines += [
        "---",
        "",
        "## Đáp án",
        "",
    ]

    for q in result.questions:
        type_label = TYPE_LABELS.get(q.q_type, q.q_type)
        ans_str = str(q.answer)
        if q.q_type == "true_false":
            ans_str = "Đúng" if q.answer else "Sai"

        lines.append(f"**Câu {q.index}** ({type_label}):")
        lines.append(f"- Đáp án: **{ans_str}**")

        if q.rubric:
            lines.append(f"- Gợi ý chấm: {q.rubric}")

        lines.append(f"- Nguồn: Trang {q.source_page}")
        lines.append("")

    return "\n".join(lines)