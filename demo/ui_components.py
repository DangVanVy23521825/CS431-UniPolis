"""
ui_components.py — Tất cả render functions tách khỏi app logic.
Mỗi hàm nhận dữ liệu thuần, không đọc session state trực tiếp.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import streamlit as st
from langchain_core.documents import Document

from config import DEMO_PIPELINE, PRESET_QUESTIONS, SUBJECT_CATALOGUE
from utils import compute_confidence, format_elapsed


# ── CSS Loader ─────────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Đọc styles.css và inject vào Streamlit."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Fallback: inline minimal CSS nếu file không tồn tại
        st.markdown(
            "<style>"
            "#MainMenu{visibility:hidden;}footer{visibility:hidden;}"
            "header{visibility:hidden;}"
            ".main .block-container{max-width:900px!important;}"
            "</style>",
            unsafe_allow_html=True,
        )


# ── Hero section ───────────────────────────────────────────────────────────────
def render_hero(subject: str, cfg: dict) -> None:
    """Render tiêu đề trang với icon môn học."""
    icon = cfg.get("icon", "📚")
    st.markdown(
        f"""
        <div class="rag-hero">
          <h1>{icon} RAG Giáo trình Lý luận Chính trị</h1>
          <p class="subtitle">Hệ thống hỏi đáp thông minh dựa trên giáo trình chính thống</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Status bar ─────────────────────────────────────────────────────────────────
def render_status_bar(subject: str, top_k: int) -> None:
    """Hiển thị thanh trạng thái cấu hình hiện tại bên trên khung chat."""
    subject_cfg = SUBJECT_CATALOGUE.get(subject, {})
    icon = subject_cfg.get("icon", "📖")
    pipeline_label = DEMO_PIPELINE["label"]
    pipeline_desc = DEMO_PIPELINE["desc"]

    st.markdown(
        f"""
        <div class="status-bar">
          <span class="status-chip">{icon} {subject}</span>
          <span class="status-chip">🚀 {pipeline_label}</span>
          <span class="status-chip">🔍 Top-{top_k}</span>
          <span class="status-chip" title="{pipeline_desc}">⚙️ {pipeline_desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_unsupported_subject(subject: str) -> None:
    """Hiển thị khi môn học chưa được hỗ trợ trong demo."""
    st.markdown(
        f"""
        <div class="error-box">
          <strong>ℹ️ Môn học hiện chưa được hỗ trợ</strong><br>
          <small>
            "{subject}" chưa có dữ liệu cho pipeline demo.
            Vui lòng chọn một trong ba môn: Lịch sử Đảng, Pháp luật đại cương, Triết học Mác-Lênin.
          </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar sections ───────────────────────────────────────────────────────────
def sidebar_section_label(text: str) -> None:
    st.markdown(
        f'<p class="sidebar-section-label">{text}</p>',
        unsafe_allow_html=True,
    )


# ── Empty state ────────────────────────────────────────────────────────────────
def render_empty_state(subject: str, presets: list[str]) -> Optional[str]:
    """
    Hiển thị empty state với preset questions.
    Trả về câu hỏi được chọn (nếu có) hoặc None.
    """
    cfg = SUBJECT_CATALOGUE.get(subject, {})
    icon = cfg.get("icon", "📚")

    st.markdown(
        f"""
        <div class="empty-state">
          <div class="icon">{icon}</div>
          <h3>Bắt đầu hỏi về {subject}</h3>
          <p>Chọn một câu hỏi mẫu bên dưới hoặc nhập câu hỏi của bạn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if presets:
        st.markdown("**💡 Câu hỏi gợi ý:**")
        # Chia preset thành 2 cột
        cols = st.columns(2)
        for i, q in enumerate(presets):
            with cols[i % 2]:
                if st.button(q, key=f"preset_{i}", use_container_width=True):
                    return q

    return None


# ── Evidence panel ──────────────────────────────────────────────────────────────
def render_evidence(docs: List[Document], subject: str) -> None:
    """Render evidence với tab, page badge, và nội dung đoạn văn."""
    if not docs:
        return

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    with st.expander(f"📎 **Nguồn trích dẫn** ({len(docs)} đoạn)", expanded=False):
        tabs_labels = [
            f"Đoạn {i+1} · tr.{_page_display(d)}"
            for i, d in enumerate(docs)
        ]
        tabs = st.tabs(tabs_labels)
        for tab, doc in zip(tabs, docs):
            with tab:
                page = _page_display(doc)
                st.markdown(
                    f"""
                    <div class="evidence-card">
                      <span class="page-badge">📄 {subject} · Trang {page}</span>
                      <div>{doc.page_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _page_display(doc: Document) -> str:
    """Chuyển metadata page thành số trang 1-indexed."""
    page = doc.metadata.get("page", "?")
    try:
        return str(int(page) + 1)
    except (ValueError, TypeError):
        return str(page)


# ── Stats row ───────────────────────────────────────────────────────────────────
def render_stats_row(elapsed: float, docs: List[Document], subject: str, model: str) -> None:
    """Hiển thị thống kê nhỏ: thời gian, số nguồn, confidence."""
    confidence = compute_confidence(docs)
    conf_class = {
        "Cao": "confidence-high",
        "Trung bình": "confidence-medium",
        "Thấp": "confidence-low",
    }.get(confidence["label"], "confidence-medium")

    elapsed_str = format_elapsed(elapsed)
    n = confidence["n_sources"]

    st.markdown(
        f"""
        <div class="stats-row">
          <span class="stat-item">⏱ {elapsed_str}</span>
          <span class="stat-item">📄 {n} nguồn</span>
          <span class="stat-item">🧠 {model}</span>
          <span class="confidence-badge {conf_class}">
            {confidence["icon"]} Độ tin cậy: {confidence["label"]}
          </span>
        </div>
        <p style="font-size:0.7rem;opacity:0.4;margin-top:0.2rem;">
          * Độ tin cậy dựa trên số đoạn ngữ cảnh truy xuất — chỉ mang tính tham khảo
        </p>
        """,
        unsafe_allow_html=True,
    )


# ── Error states ───────────────────────────────────────────────────────────────
def render_db_missing_error(subject: str, chroma_dir: str) -> None:
    """Hiển thị khi chưa có DB cho môn này."""
    st.markdown(
        f"""
        <div class="error-box">
          <strong>⚠️ Chưa có dữ liệu cho môn "{subject}"</strong><br>
          <small>
            Hãy chạy lệnh sau để ingest PDF vào vector store:<br>
            <code>python ingest.py --chroma_dir {chroma_dir}</code>
          </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_engine_error(exc: Exception, context: str = "") -> None:
    """Hiển thị lỗi engine thân thiện với người dùng."""
    exc_type = type(exc).__name__
    msg = str(exc)
    hint = _error_hint(exc_type, msg)

    st.markdown(
        f"""
        <div class="error-box">
          <strong>❌ Đã xảy ra lỗi</strong>{f" khi {context}" if context else ""}<br>
          <small><code>{exc_type}: {msg[:200]}</code></small>
          {f'<br><small>💡 {hint}</small>' if hint else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _error_hint(exc_type: str, msg: str) -> str:
    """Gợi ý xử lý dựa trên loại lỗi."""
    msg_lower = msg.lower()
    if "api_key" in msg_lower or "authentication" in msg_lower:
        return "Kiểm tra OPENAI_API_KEY trong file .env"
    if "rate limit" in msg_lower:
        return "API đang bị rate limit — thử lại sau vài giây"
    if "connection" in msg_lower or "timeout" in msg_lower:
        return "Lỗi kết nối mạng — kiểm tra internet"
    if "cuda" in msg_lower or "device" in msg_lower:
        return "Lỗi thiết bị GPU/CPU — thử chuyển sang embedding model khác"
    if "bm25" in msg_lower or "pickle" in msg_lower:
        return "BM25 index bị hỏng — chạy lại ingest.py"
    return ""


# ── Skeleton loading ────────────────────────────────────────────────────────────
def render_skeleton() -> None:
    """Hiển thị skeleton loading placeholder."""
    st.markdown(
        """
        <div style="padding: 0.5rem 0;">
          <div class="skeleton-line long"></div>
          <div class="skeleton-line medium"></div>
          <div class="skeleton-line long"></div>
          <div class="skeleton-line short"></div>
          <div class="skeleton-line medium"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Export buttons ──────────────────────────────────────────────────────────────
def render_export_buttons(md_content: str, txt_content: str) -> None:
    """Render nút download export markdown và txt."""
    now_str = datetime.now_str()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Export Markdown",
            data=md_content,
            file_name=f"rag_conversation_{now_str}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📄 Export Text",
            data=txt_content,
            file_name=f"rag_conversation_{now_str}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Workaround: datetime helper (avoid import cycle) ──────────────────────────
class datetime:
    @staticmethod
    def now_str() -> str:
        from datetime import datetime as _dt
        return _dt.now().strftime("%Y%m%d_%H%M%S")
