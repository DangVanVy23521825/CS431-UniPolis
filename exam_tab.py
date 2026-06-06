"""
exam_tab.py — Tab "Tạo đề thi" cho Streamlit app.

Cách dùng trong app.py:
    from exam_tab import render_exam_tab
    # Trong main(), sau khi load engine:
    tab_chat, tab_exam = st.tabs(["💬 Hỏi đáp", "📝 Tạo đề thi"])
    with tab_chat:
        # ... code chat hiện tại ...
    with tab_exam:
        render_exam_tab(engine, subject, pipeline_mode)
"""

from __future__ import annotations

import json
import re
from typing import Optional

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import SUBJECT_CATALOGUE
from exam_generator import (
    DIFF_LABELS,
    TYPE_LABELS,
    ExamGenerator,
    ExamQuestion,
    ExamResult,
    export_exam_markdown,
)
from utils import log_error


# ── Cache ExamGenerator (dùng lại engine đã cache) ────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_exam_generator(engine_id: str, _engine) -> ExamGenerator:
    return ExamGenerator(rag_engine=_engine)


# ── Attempt state + grading helpers ────────────────────────────────────────────
def _reset_exam_attempt_state() -> None:
    st.session_state.exam_submitted = False
    st.session_state.exam_grading = None
    for k in list(st.session_state.keys()):
        if k.startswith("exam_answer_"):
            del st.session_state[k]


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _text_similarity(user_text: str, ref_text: str) -> float:
    user_tokens = set(_normalize_text(user_text).split())
    ref_tokens = set(_normalize_text(ref_text).split())
    if not user_tokens or not ref_tokens:
        return 0.0
    overlap = len(user_tokens & ref_tokens)
    return overlap / max(1, len(ref_tokens))


@st.cache_resource(show_spinner=False)
def _get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=300,
    )


def _heuristic_open_grading(user_answer: object, reference: object) -> dict:
    sim = _text_similarity(str(user_answer), str(reference))
    if sim >= 0.7:
        earned = 1.0
        fb = "Rất sát đáp án."
    elif sim >= 0.45:
        earned = 0.5
        fb = "Đúng một phần, cần đầy đủ hơn."
    else:
        earned = 0.0
        fb = "Chưa sát đáp án gợi ý."
    return {
        "is_correct": earned >= 1.0,
        "earned": earned,
        "max_score": 1.0,
        "feedback": fb,
    }


def _grade_open_answer_with_llm(q: ExamQuestion, user_answer: str) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là giám khảo công tâm. "
                "Hãy chấm câu trả lời của sinh viên theo thang 0/0.5/1 dựa trên câu hỏi, đáp án chuẩn và rubric. "
                "Trả về JSON thuần: {\"score\":0|0.5|1, \"feedback\":\"...\"}. "
                "Feedback tối đa 1 câu, tiếng Việt.",
            ),
            (
                "human",
                "CÂU HỎI:\n{question}\n\n"
                "ĐÁP ÁN CHUẨN (nếu có):\n{answer_key}\n\n"
                "RUBRIC (nếu có):\n{rubric}\n\n"
                "CÂU TRẢ LỜI SINH VIÊN:\n{student_answer}",
            ),
        ]
    )
    chain = prompt | _get_judge_llm() | StrOutputParser()
    raw = chain.invoke(
        {
            "question": q.question or "",
            "answer_key": str(q.answer or ""),
            "rubric": str(q.rubric or ""),
            "student_answer": user_answer,
        }
    )
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    data = json.loads(clean)
    score = data.get("score", 0)
    try:
        score_f = float(score)
    except Exception:
        score_f = 0.0
    if score_f >= 0.75:
        score_f = 1.0
    elif score_f >= 0.25:
        score_f = 0.5
    else:
        score_f = 0.0
    feedback = str(data.get("feedback", "")).strip() or "Đã chấm theo rubric."
    return {
        "is_correct": score_f >= 1.0,
        "earned": score_f,
        "max_score": 1.0,
        "feedback": feedback,
    }


def _grade_single_question(q: ExamQuestion, user_answer: object) -> dict:
    base = {
        "is_correct": False,
        "earned": 0.0,
        "max_score": 1.0,
        "feedback": "Chưa có câu trả lời.",
    }
    if user_answer in (None, "", "__none__"):
        return base

    if q.q_type == "mcq":
        is_correct = str(user_answer) == str(q.answer)
        return {
            "is_correct": is_correct,
            "earned": 1.0 if is_correct else 0.0,
            "max_score": 1.0,
            "feedback": "Đúng." if is_correct else f"Sai. Đáp án đúng: {q.answer}",
        }

    if q.q_type == "true_false":
        gt = "true" if bool(q.answer) else "false"
        ua = "true" if str(user_answer).lower() in ("true", "đúng", "dung", "1") else "false"
        is_correct = ua == gt
        return {
            "is_correct": is_correct,
            "earned": 1.0 if is_correct else 0.0,
            "max_score": 1.0,
            "feedback": "Đúng." if is_correct else f"Sai. Đáp án đúng: {'Đúng' if bool(q.answer) else 'Sai'}",
        }

    if q.q_type in ("short_answer", "scenario"):
        try:
            return _grade_open_answer_with_llm(q, str(user_answer))
        except Exception as exc:
            log_error("exam_grade.llm_judge", exc)
            reference = q.answer or q.rubric or ""
            return _heuristic_open_grading(user_answer, reference)

    # fill_blank: giữ heuristic nhẹ để phản hồi nhanh
    reference = q.answer or q.rubric or ""
    return _heuristic_open_grading(user_answer, reference)


def _grade_exam(result: ExamResult) -> dict:
    details: dict[int, dict] = {}
    total = 0.0
    max_total = float(len(result.questions))
    for q in result.questions:
        user_answer = st.session_state.get(f"exam_answer_{q.index}")
        detail = _grade_single_question(q, user_answer)
        details[q.index] = detail
        total += detail["earned"]
    return {
        "score": round(total, 2),
        "max_score": round(max_total, 2),
        "percent": round((total / max_total) * 100, 1) if max_total > 0 else 0.0,
        "details": details,
    }


# ── Quality score display ──────────────────────────────────────────────────────
def _quality_color(score: float) -> str:
    if score >= 0.85:
        return "#16A34A"   # green
    if score >= 0.70:
        return "#D97706"   # amber
    return "#DC2626"       # red


def _quality_label(score: float) -> str:
    if score >= 0.85:
        return "Tốt"
    if score >= 0.70:
        return "Khá"
    return "Yếu"


# ── Render một câu hỏi ─────────────────────────────────────────────────────────
def _render_question(
    q: ExamQuestion,
    show_answer: bool = False,
    submitted: bool = False,
    grading_detail: Optional[dict] = None,
) -> None:
    """Render một ExamQuestion trong expander."""
    type_label = TYPE_LABELS.get(q.q_type, q.q_type)
    diff_label = DIFF_LABELS.get(q.difficulty, q.difficulty)
    q_color = _quality_color(q.quality_score)
    q_label = _quality_label(q.quality_score)

    # Header row
    col_meta, col_score = st.columns([3, 1])
    with col_meta:
        st.markdown(
            f"**Câu {q.index}** &nbsp; "
            f"`{type_label}` &nbsp; "
            f"`{diff_label}` &nbsp; "
            f"📄 Trang {q.source_page}",
            unsafe_allow_html=True,
        )
    with col_score:
        st.markdown(
            f'<span style="color:{q_color};font-size:0.8rem;font-weight:600;">'
            f"● {q.quality_score:.2f} {q_label}</span>",
            unsafe_allow_html=True,
        )

    # Question text
    st.markdown(q.question)

    # MCQ options
    if q.q_type == "mcq" and q.options:
        for letter, text in q.options.items():
            is_correct = show_answer and str(q.answer) == letter
            prefix = "✅ " if is_correct else f"**{letter}.** "
            style = "color:#16A34A;font-weight:600;" if is_correct else ""
            st.markdown(
                f'<span style="{style}">{prefix}{text}</span>',
                unsafe_allow_html=True,
            )
        choices = ["__none__"] + list(q.options.keys())
        st.selectbox(
            "Chọn đáp án",
            options=choices,
            format_func=lambda x: "— Chọn đáp án —" if x == "__none__" else f"{x}. {q.options.get(x, '')}",
            key=f"exam_answer_{q.index}",
            disabled=submitted,
        )
    elif q.q_type == "true_false":
        st.selectbox(
            "Chọn đáp án",
            options=["__none__", "true", "false"],
            format_func=lambda x: {
                "__none__": "— Chọn đáp án —",
                "true": "Đúng",
                "false": "Sai",
            }[x],
            key=f"exam_answer_{q.index}",
            disabled=submitted,
        )
    elif q.q_type == "fill_blank":
        st.text_input(
            "Điền đáp án",
            key=f"exam_answer_{q.index}",
            disabled=submitted,
            placeholder="Nhập từ/cụm từ điền khuyết",
        )
    else:
        st.text_area(
            "Câu trả lời",
            key=f"exam_answer_{q.index}",
            disabled=submitted,
            placeholder="Nhập câu trả lời của bạn",
            height=100,
        )

    # Answer (nếu show)
    if show_answer:
        if q.q_type == "true_false":
            ans_display = "✅ Đúng" if q.answer else "❌ Sai"
            st.success(f"Đáp án: {ans_display}")
        elif q.q_type in ("fill_blank", "short_answer", "scenario"):
            st.info(f"💡 Đáp án gợi ý: {q.answer or q.rubric or 'N/A'}")
        # MCQ answer already highlighted above

    if submitted and grading_detail:
        if grading_detail["earned"] >= 1.0:
            st.success(f"✔️ {grading_detail['feedback']} (+1.0)")
        elif grading_detail["earned"] > 0:
            st.warning(f"➖ {grading_detail['feedback']} (+{grading_detail['earned']:.1f})")
        else:
            st.error(f"✖️ {grading_detail['feedback']} (+0.0)")

    st.markdown(
        f'<p style="font-size:0.72rem;color:gray;margin-top:0.4rem;">'
        f'Nguồn: {q.source_page} · Quality hint: {q.quality_score:.2f} (heuristic)</p>',
        unsafe_allow_html=True,
    )
    st.divider()


# ── Main tab renderer ──────────────────────────────────────────────────────────
def render_exam_tab(engine, subject: str, pipeline_mode: str) -> None:
    """
    Render toàn bộ tab "Tạo đề thi".
    Gọi từ app.py sau khi engine đã được load.
    """
    st.markdown("### 📝 Tạo đề thi từ giáo trình")
    st.caption(
        "Câu hỏi được sinh có kèm nguồn trang. "
        "Quality score chỉ mang tính tham khảo (heuristic)."
    )

    # ── Config form ────────────────────────────────────────────────────────────
    with st.expander("⚙️ Cấu hình đề thi", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Chủ đề cụ thể (để trống = tổng hợp)",
                placeholder="VD: Cách mạng tháng Tám 1945",
                key="exam_topic",
            )
            n_questions = st.slider(
                "Số câu hỏi",
                min_value=3,
                max_value=15,
                value=6,
                step=1,
                key="exam_n_questions",
            )

        with col2:
            difficulty = st.radio(
                "Độ khó",
                options=["easy", "medium", "hard", "mixed"],
                format_func=lambda x: {"easy": "Dễ", "medium": "Vừa", "hard": "Khó", "mixed": "Hỗn hợp"}[x],
                index=1,
                horizontal=True,
                key="exam_difficulty",
            )
            show_answers = st.toggle("Hiển thị đáp án", value=True, key="exam_show_answers")

        st.markdown("**Tỉ lệ loại câu hỏi** (tổng không cần = 100%)")
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            pct_mcq = st.number_input("MCQ %", min_value=0, max_value=100, value=50, step=5, key="pct_mcq")
        with r_col2:
            pct_tf = st.number_input("Đúng/Sai %", min_value=0, max_value=100, value=20, step=5, key="pct_tf")
        with r_col3:
            pct_fill = st.number_input("Điền khuyết %", min_value=0, max_value=100, value=15, step=5, key="pct_fill")
        with r_col4:
            pct_short = st.number_input("Tự luận %", min_value=0, max_value=100, value=15, step=5, key="pct_short")

    type_ratios = {
        "mcq": pct_mcq,
        "true_false": pct_tf,
        "fill_blank": pct_fill,
        "short_answer": pct_short,
    }

    # ── Generate button ────────────────────────────────────────────────────────
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        generate_clicked = st.button(
            "🎯 Tạo đề thi",
            type="primary",
            use_container_width=True,
            key="exam_generate_btn",
        )

    # ── Session state for exam result ──────────────────────────────────────────
    if "exam_result" not in st.session_state:
        st.session_state.exam_result = None
    if "exam_submitted" not in st.session_state:
        st.session_state.exam_submitted = False
    if "exam_grading" not in st.session_state:
        st.session_state.exam_grading = None

    # ── Run generation ─────────────────────────────────────────────────────────
    if generate_clicked:
        generator = _get_exam_generator(id(engine), engine)

        with st.spinner(f"🤔 Đang tạo {n_questions} câu hỏi..."):
            try:
                result: ExamResult = generator.generate(
                    subject=subject,
                    n_questions=n_questions,
                    topic=topic,
                    difficulty=difficulty,
                    type_ratios=type_ratios,
                )
                st.session_state.exam_result = result
                _reset_exam_attempt_state()
            except Exception as exc:
                log_error("exam_generator.generate", exc)
                st.error(f"❌ Lỗi khi tạo đề: {exc}")
                return

    # ── Render result ──────────────────────────────────────────────────────────
    result: Optional[ExamResult] = st.session_state.exam_result

    if result is None:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;opacity:0.45;">
              <div style="font-size:2.5rem;">📋</div>
              <p style="margin-top:0.75rem;font-size:0.9rem;">
                Chưa có đề thi nào. Cấu hình và nhấn "Tạo đề thi".
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Refused state
    if result.refused:
        st.warning(
            f"⚠️ **Không thể tạo đề:** {result.refuse_reason}\n\n"
            "Gợi ý: Thử chủ đề rộng hơn, hoặc kiểm tra dữ liệu đã ingest."
        )
        return

    # Stats bar
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Câu hỏi", len(result.questions))
    s2.metric("Nguồn trang", result.n_sources)
    s3.metric("Avg quality", f"{result.avg_quality:.2f}")
    s4.metric("Thời gian", f"{result.elapsed:.1f}s")

    st.markdown("---")

    # Export buttons
    md_content = export_exam_markdown(result)
    ex1, ex2, ex3 = st.columns([1, 1, 4])
    with ex1:
        st.download_button(
            "📥 Markdown",
            data=md_content,
            file_name=f"de_thi_{subject[:20].replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            "📄 Text",
            data=md_content,
            file_name=f"de_thi_{subject[:20].replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")

    # Submit + grading
    submit_col, score_col = st.columns([1, 2])
    with submit_col:
        submit_clicked = st.button(
            "✅ Nộp bài & chấm điểm",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.exam_submitted,
            key="exam_submit_btn",
        )
    if submit_clicked:
        st.session_state.exam_grading = _grade_exam(result)
        st.session_state.exam_submitted = True
        st.rerun()

    with score_col:
        grading = st.session_state.exam_grading
        if grading:
            st.info(
                f"Điểm hiện tại: **{grading['score']}/{grading['max_score']}** "
                f"({grading['percent']}%)"
            )

    reveal_answers = show_answers and st.session_state.exam_submitted
    if show_answers and not st.session_state.exam_submitted:
        st.caption("Nộp bài trước để xem đáp án.")

    # Questions
    for q in result.questions:
        detail = None
        if st.session_state.exam_grading:
            detail = st.session_state.exam_grading["details"].get(q.index)
        _render_question(
            q,
            show_answer=reveal_answers,
            submitted=st.session_state.exam_submitted,
            grading_detail=detail,
        )

    # Clear button
    if st.button("🗑️ Xóa đề thi", key="exam_clear"):
        st.session_state.exam_result = None
        _reset_exam_attempt_state()
        st.rerun()