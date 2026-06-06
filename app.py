"""
app.py — RAG Giáo trình Lý luận Chính trị (bản có Tạo đề thi)

Thay đổi so với bản upgrade trước:
  - Thêm tab "Tạo đề thi" tích hợp exam_tab.py
  - Cập nhật màu sắc edu scheme (đỏ/vàng)
  - Hero mới với layout horizontal

Chạy: streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import (
    PIPELINE_DESC,
    PIPELINE_MODES,
    PRESET_QUESTIONS,
    SUBJECT_CATALOGUE,
)
from ui_components import (
    inject_css,
    render_db_missing_error,
    render_empty_state,
    render_engine_error,
    render_evidence,
    render_export_buttons,
    render_skeleton,
    render_stats_row,
    render_status_bar,
    sidebar_section_label,
)
from utils import (
    export_conversation_markdown,
    export_conversation_txt,
    log_error,
    log_query,
)
from exam_tab import render_exam_tab


def _import_rag_engine():
    from rag_engine import HistoryRAG
    return HistoryRAG


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Giáo trình Lý luận Chính trị",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ── Session state ──────────────────────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "messages": [],
        "pending_question": None,
        "last_config": {},
        "regenerating": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _detect_config_change(subject, pipeline_mode, top_k) -> bool:
    cur = {"subject": subject, "pipeline_mode": pipeline_mode, "top_k": top_k}
    changed = st.session_state.last_config and st.session_state.last_config != cur
    st.session_state.last_config = cur
    return changed


def _resolve_subject_store(subject_cfg: dict, pipeline_mode_key: str) -> tuple[str, str]:
    """Use the vector store built with the same embedding model as the selected mode."""
    mode_key = PIPELINE_MODES[pipeline_mode_key].get("key")
    if mode_key == "quality_vn_bi":
        return subject_cfg["chroma_dir_vn_bi_ft"], subject_cfg["collection_vn_bi_ft"]
    return subject_cfg["chroma_dir"], subject_cfg["collection"]


def _clear_history():
    st.session_state.messages = []
    st.session_state.pending_question = None


# ── Engine loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải mô hình và vector store…")
def load_engine(chroma_dir, collection, pipeline_mode_key, top_k):
    """Tải HistoryRAG theo pipeline mode đã chọn."""
    mode_cfg = PIPELINE_MODES[pipeline_mode_key]
    try:
        HistoryRAG = _import_rag_engine()
        return HistoryRAG(
            chroma_dir=chroma_dir,
            collection_name=collection,
            embedding_model_key=mode_cfg["embedding"],
            rerank_top_k=top_k,
            retriever_mode=mode_cfg["retriever_mode"],
            use_rerank=mode_cfg["use_rerank"],
            reranker_model=mode_cfg.get("reranker_model"),
        ), None
    except FileNotFoundError as e:
        return None, f"BM25 index không tìm thấy: {e}"
    except Exception as e:
        log_error("load_engine", e)
        return None, str(e)


# ── Answer generation ──────────────────────────────────────────────────────────
def _run_ask(engine, question: str):
    t0 = time.time()
    answer, sources = engine.ask(question)
    return answer, sources, time.time() - t0


# ── Chat rendering ─────────────────────────────────────────────────────────────
def _render_chat_history(subject: str) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if "elapsed" in msg and "sources" in msg:
                    render_stats_row(msg.get("elapsed", 0), msg.get("sources", []), subject, msg.get("model", ""))
                if msg.get("sources"):
                    render_evidence(msg["sources"], subject)


def _get_last_user_question() -> Optional[str]:
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            return msg["content"]
    return None


def _truncate_to_last_user_message() -> None:
    msgs = st.session_state.messages
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()


# ── Hero ───────────────────────────────────────────────────────────────────────
def _render_hero(subject: str, cfg: dict) -> None:
    icon = cfg.get("icon", "📚")
    st.markdown(
        f"""
        <div class="rag-hero">
          <div>
            <h1>{icon} RAG Giáo trình Lý luận Chính trị</h1>
            <p class="subtitle">Trợ lý hỏi đáp thông minh · BM25 RAG (No-rerank) · GPT-4o-mini</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Cấu hình")

        sidebar_section_label("Môn học")
        subject = st.selectbox(
            "Giáo trình",
            list(SUBJECT_CATALOGUE.keys()),
            label_visibility="collapsed",
        )

        sidebar_section_label("Pipeline Mode")
        pipeline_mode = st.selectbox(
            "Pipeline Mode",
            list(PIPELINE_MODES.keys()),
            index=1,  # mặc định: ⚖️ Balance
            label_visibility="collapsed",
        )
        # Hiển thị mô tả chi tiết của mode đang chọn
        st.caption(PIPELINE_MODES[pipeline_mode]["detail"])

        sidebar_section_label("Tham số truy xuất")
        default_top_k = PIPELINE_MODES[pipeline_mode]["default_top_k"]
        top_k = st.slider("Top-K", 1, 10, default_top_k, 1)

        st.divider()
        sidebar_section_label("Chat actions")
        c1, c2 = st.columns(2)
        with c1:
            clear_btn = st.button("🗑️ Xóa chat", use_container_width=True)
        with c2:
            regen_btn = st.button("🔄 Tái tạo", use_container_width=True)

        if st.session_state.get("messages"):
            sidebar_section_label("Export")
            md = export_conversation_markdown(st.session_state.messages, subject, pipeline_mode)
            txt = export_conversation_txt(st.session_state.messages)
            render_export_buttons(md, txt)

        with st.expander("ℹ️ Pipeline", expanded=False):
            st.caption(PIPELINE_DESC)

        if clear_btn:
            _clear_history()
            st.toast("🗑️ Đã xóa lịch sử", icon="✅")
            st.rerun()

        if regen_btn:
            last_q = _get_last_user_question()
            if last_q:
                _truncate_to_last_user_message()
                st.session_state.pending_question = last_q
                st.session_state.regenerating = True
                st.rerun()
            else:
                st.toast("Chưa có câu hỏi nào để tái tạo", icon="⚠️")

    return subject, pipeline_mode, top_k


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session_state()

    subject, pipeline_mode, top_k = _render_sidebar()
    subject_cfg = SUBJECT_CATALOGUE[subject]
    chroma_dir, collection = _resolve_subject_store(subject_cfg, pipeline_mode)

    if _detect_config_change(subject, pipeline_mode, top_k):
        st.toast(f"⚙️ Đã đổi cấu hình: {subject} · {pipeline_mode}", icon="🔄")

    _render_hero(subject, subject_cfg)

    # DB check
    db_path = Path(chroma_dir)
    if not (db_path.exists() and any(db_path.iterdir())):
        render_db_missing_error(subject, chroma_dir)
        st.stop()

    # Engine — cache key gồm cả pipeline_mode để tự reload khi đổi mode
    engine, engine_error = load_engine(chroma_dir, collection, pipeline_mode, top_k)
    if engine_error or engine is None:
        render_engine_error(Exception(engine_error or "Unknown"), context="tải engine")
        st.stop()

    # chat_input phải ở NGOÀI tabs (Streamlit constraint)
    user_input = st.chat_input(f"Nhập câu hỏi về {subject}…")

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_chat, tab_exam = st.tabs(["💬 Hỏi đáp", "📝 Tạo đề thi"])

    # ── Tab Chat ─────────────────────────────────────────────────────────────
    with tab_chat:
        render_status_bar(subject, pipeline_mode, top_k)

        if st.session_state.regenerating:
            st.info("🔄 Đang tái tạo câu trả lời…")

        if st.session_state.messages:
            _render_chat_history(subject)
        else:
            presets = PRESET_QUESTIONS.get(subject, [])
            chosen = render_empty_state(subject, presets)
            if chosen:
                st.session_state.pending_question = chosen
                st.rerun()

    # ── Tab Exam ──────────────────────────────────────────────────────────────
    with tab_exam:
        render_exam_tab(engine, subject, pipeline_mode)

    # ── Process chat input (outside tabs) ─────────────────────────────────────
    question: Optional[str] = user_input or st.session_state.pending_question
    if st.session_state.pending_question:
        st.session_state.pending_question = None

    if not question:
        return

    regen_flag = st.session_state.regenerating
    st.session_state.regenerating = False

    # Re-render chat tab content then append new exchange
    if not regen_flag:
        with tab_chat:
            with st.chat_message("user"):
                st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

    with tab_chat:
        with st.chat_message("assistant"):
            skeleton_ph = st.empty()
            with skeleton_ph.container():
                render_skeleton()
            try:
                answer, sources, elapsed = _run_ask(engine, question)
                log_query(subject, pipeline_mode, question, elapsed, len(sources))
            except Exception as exc:
                skeleton_ph.empty()
                log_error("engine.ask", exc)
                render_engine_error(exc, context="truy vấn")
                st.stop()

            skeleton_ph.empty()
            st.markdown(answer)
            render_stats_row(elapsed, sources, subject, pipeline_mode)
            render_evidence(sources, subject)

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "elapsed": elapsed, "model": pipeline_mode,
    })


if __name__ == "__main__":
    main()
