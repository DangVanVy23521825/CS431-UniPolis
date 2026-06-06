"""
utils.py — Utility helpers: logging, export, confidence hint, session state.
"""

from __future__ import annotations

import logging
import logging.handlers
import textwrap
from datetime import datetime
from typing import List

from langchain_core.documents import Document

from config import (
    CONFIDENCE_THRESHOLDS,
    LOG_BACKUP_COUNT,
    LOG_FILE,
    LOG_MAX_BYTES,
)


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str = "rag_app") -> logging.Logger:
    """
    Trả về logger có rotating file handler.
    Không log API key hay thông tin nhạy cảm.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stream handler (console) — INFO+
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)

    return logger


logger = get_logger()


def log_query(subject: str, model: str, question: str, elapsed: float, n_sources: int) -> None:
    """Log một truy vấn (không log nội dung câu trả lời hay API key)."""
    logger.info(
        "QUERY | subject=%s | model=%s | elapsed=%.2fs | sources=%d | q_len=%d",
        subject,
        model,
        elapsed,
        n_sources,
        len(question),
    )


def log_error(context: str, exc: Exception) -> None:
    """Log lỗi với context string."""
    logger.error("ERROR | context=%s | %s: %s", context, type(exc).__name__, exc)


# ── Confidence hint ───────────────────────────────────────────────────────────

def compute_confidence(docs: List[Document]) -> dict:
    """
    Heuristic đơn giản: confidence dựa trên số nguồn truy xuất được.
    Minh bạch với người dùng — không giả vờ là ML score.
    """
    n = len(docs)
    if n >= CONFIDENCE_THRESHOLDS["high"]["min_sources"]:
        tier = CONFIDENCE_THRESHOLDS["high"]
    elif n >= CONFIDENCE_THRESHOLDS["medium"]["min_sources"]:
        tier = CONFIDENCE_THRESHOLDS["medium"]
    else:
        tier = CONFIDENCE_THRESHOLDS["low"]

    return {
        "label": tier["label"],
        "color": tier["color"],
        "icon": tier["icon"],
        "n_sources": n,
        "note": "Dựa trên số đoạn ngữ cảnh truy xuất được (heuristic)",
    }


# ── Export ────────────────────────────────────────────────────────────────────

def export_conversation_markdown(messages: list, subject: str, pipeline_mode: str) -> str:
    """
    Chuyển lịch sử hội thoại thành chuỗi Markdown để export.
    Bao gồm metadata và nguồn trích dẫn.
    """
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    lines: list[str] = [
        "# 📚 Hội thoại RAG — Giáo trình Lý luận Chính trị",
        "",
        f"- **Môn học:** {subject}",
        f"- **Pipeline mode:** {pipeline_mode}",
        f"- **Thời gian export:** {now}",
        "",
        "---",
        "",
    ]

    for i, msg in enumerate(messages, 1):
        role = "👤 Người dùng" if msg["role"] == "user" else "🤖 Trợ lý"
        lines.append(f"### {role}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")

        # Gắn thông tin nguồn nếu có
        if msg["role"] == "assistant" and msg.get("sources"):
            sources: List[Document] = msg["sources"]
            lines.append(f"**📎 Nguồn trích dẫn ({len(sources)} đoạn):**")
            for j, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                try:
                    page_display = int(page) + 1
                except (ValueError, TypeError):
                    page_display = page
                # Chỉ lấy 120 ký tự đầu mỗi đoạn để export gọn
                snippet = textwrap.shorten(doc.page_content, width=120, placeholder="…")
                lines.append(f"  {j}. Trang {page_display}: _{snippet}_")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_conversation_txt(messages: list) -> str:
    """Export hội thoại dạng plain text."""
    lines: list[str] = []
    for msg in messages:
        prefix = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"[{prefix}]")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


# ── Format helpers ────────────────────────────────────────────────────────────

def format_elapsed(seconds: float) -> str:
    """Định dạng thời gian xử lý dễ đọc."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"
