"""
config.py — Cấu hình tập trung cho toàn bộ ứng dụng RAG.
"""

from __future__ import annotations

# ── Pipeline modes ────────────────────────────────────────────────────────────
# 3 chế độ pipeline, mỗi chế độ xác định toàn bộ cấu hình retrieval:
#   Fast    : BM25 · No Rerank          → nhanh nhất
#   Balance : BM25 · No Rerank          → cân bằng (top_k lớn hơn)
#   Quality : Dense MMR · BGE Reranker  → chất lượng cao nhất
PIPELINE_MODES: dict[str, dict] = {
    "⚡ Fast": {
        "key": "fast",
        "embedding": "BGE-M3 (Local)",
        "retriever_mode": "bm25",
        "default_top_k": 3,
        "use_rerank": False,
        "reranker_model": None,
        "desc": "Recursive-Char · BGE-M3 · BM25 · Không rerank",
        "detail": "Tốc độ tối ưu — BM25 trả về 3 đoạn tốt nhất, không qua reranker.",
    },
    "⚖️ Balance": {
        "key": "balance",
        "embedding": "BGE-M3 (Local)",
        "retriever_mode": "bm25",
        "default_top_k": 5,
        "use_rerank": False,
        "reranker_model": None,
        "desc": "Recursive-Char · BGE-M3 · BM25 · Không rerank",
        "detail": "Cân bằng tốc độ & chất lượng — BM25 trả về 5 đoạn, không qua reranker.",
    },
    "🎯 Quality": {
        "key": "quality",
        "embedding": "BGE-M3 (Local)",
        "retriever_mode": "dense_mmr",
        "default_top_k": 5,
        "use_rerank": True,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "desc": "Recursive-Char · BGE-M3 · Dense MMR · BGE Reranker",
        "detail": "Chất lượng cao nhất — Dense MMR lấy đoạn đa dạng, CrossEncoder reranker tinh chỉnh lại.",
    },
    # Chỉ dùng sau khi Chroma đã ingest lại bằng đúng model FT (cùng chunking recursive 512/64).
    "🎯 Quality · VN Bi-Encoder (FT)": {
        "key": "quality_vn_bi",
        "embedding": "Vietnamese Bi-Encoder (FT)",
        "retriever_mode": "dense_mmr",
        "default_top_k": 5,
        "use_rerank": True,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "desc": "Recursive-Char · VN Bi-Encoder FT · Dense MMR · BGE Reranker",
        "detail": "Giống Quality nhưng dense dùng embedding fine-tune — vector store phải build bằng cùng model.",
    },
}

# ── Catalogue môn học ─────────────────────────────────────────────────────────
SUBJECT_CATALOGUE: dict[str, dict] = {
    "Lịch sử Đảng Cộng sản Việt Nam": {
        "chroma_dir": "data/chroma_db_lich_su_dang_e5_bm25",
        "collection": "rag_lich_su_dang_e5_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_lich_su_dang_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_lich_su_dang_recursive_vn_bi_ft_bm25",
        "icon": "🏛️",
        "color": "#e74c3c",
    },
    "Pháp luật đại cương": {
        "chroma_dir": "data/chroma_db_phap_luat_dai_cuong_e5_bm25",
        "collection": "rag_pldc_e5_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_phap_luat_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_phap_luat_recursive_vn_bi_ft_bm25",
        "icon": "⚖️",
        "color": "#2980b9",
    },
    "Triết học Mác-Lênin": {
        "chroma_dir": "data/chroma_db_triet_hoc_e5_bm25",
        "collection": "rag_triet_hoc_e5_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_triet_hoc_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_triet_hoc_recursive_vn_bi_ft_bm25",
        "icon": "🔬",
        "color": "#8e44ad",
    },
    "Kinh tế Chính trị Mác-Lênin": {
        "chroma_dir": "data/chroma_db_kinh_te_chinh_tri_e5_bm25",
        "collection": "rag_kinh_te_chinh_tri_e5_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_kinh_te_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_kinh_te_recursive_vn_bi_ft_bm25",
        "icon": "📊",
        "color": "#27ae60",
    },
    "Chủ nghĩa Xã hội Khoa học": {
        "chroma_dir": "data/chroma_db_chu_nghia_xa_hoi_khoa_hoc_e5_bm25",
        "collection": "rag_chu_nghia_xa_hoi_khoa_hoc_e5_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_cnxhkh_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_cnxhkh_recursive_vn_bi_ft_bm25",
        "icon": "🌐",
        "color": "#e67e22",
    },
    "Tư tưởng Hồ Chí Minh": {
        "chroma_dir": "data/chroma_db_tu_tuong_hcm_e5_bm25",
        "collection": "rag_tu_tuong_hcm_e5_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_tthcm_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_tthcm_recursive_vn_bi_ft_bm25",
        "icon": "⭐",
        "color": "#f39c12",
    },
}

# ── Preset câu hỏi mẫu theo môn ──────────────────────────────────────────────
PRESET_QUESTIONS: dict[str, list[str]] = {
    "Lịch sử Đảng Cộng sản Việt Nam": [
        "Đảng Cộng sản Việt Nam được thành lập năm nào và ở đâu?",
        "Ý nghĩa lịch sử của Cách mạng tháng Tám 1945 là gì?",
        "Đường lối đổi mới của Đảng từ năm 1986 có nội dung cốt lõi gì?",
        "Vai trò của Nguyễn Ái Quốc trong việc thành lập Đảng?",
    ],
    "Pháp luật đại cương": [
        "Nhà nước pháp quyền XHCN Việt Nam có đặc trưng gì?",
        "Phân biệt vi phạm hành chính và vi phạm hình sự?",
        "Quyền và nghĩa vụ cơ bản của công dân Việt Nam là gì?",
        "Khái niệm và đặc điểm của pháp luật là gì?",
    ],
    "Triết học Mác-Lênin": [
        "Vật chất là gì theo quan điểm của Mác-Lênin?",
        "Phép biện chứng duy vật khác gì so với phép biện chứng duy tâm?",
        "Quy luật mâu thuẫn trong phép biện chứng duy vật?",
        "Thực tiễn có vai trò gì trong nhận thức?",
    ],
    "Kinh tế Chính trị Mác-Lênin": [
        "Hàng hoá là gì? Hai thuộc tính của hàng hoá?",
        "Quy luật giá trị thặng dư hoạt động như thế nào?",
        "Kinh tế thị trường định hướng XHCN là gì?",
        "Công nghiệp hoá, hiện đại hoá ở Việt Nam có đặc điểm gì?",
    ],
    "Chủ nghĩa Xã hội Khoa học": [
        "Chủ nghĩa xã hội khoa học là gì? Đối tượng nghiên cứu?",
        "Sứ mệnh lịch sử của giai cấp công nhân là gì?",
        "Đặc trưng của chủ nghĩa xã hội theo quan điểm Mác-Lênin?",
        "Vấn đề dân tộc trong thời kỳ quá độ lên CNXH?",
    ],
    "Tư tưởng Hồ Chí Minh": [
        "Tư tưởng Hồ Chí Minh về độc lập dân tộc gắn liền với CNXH?",
        "Quan điểm của Hồ Chí Minh về Đảng Cộng sản Việt Nam?",
        "Tư tưởng Hồ Chí Minh về đại đoàn kết dân tộc?",
        "Phong cách tư duy của Hồ Chí Minh có đặc điểm gì?",
    ],
}

# ── (Legacy — giữ lại để không làm vỡ import cũ nếu có) ─────────────────────
EMBEDDING_OPTIONS: list[str] = ["BGE-M3 (Local)", "Vietnamese Bi-Encoder (FT)"]
ANSWER_MODES: dict[str, str] = {"📝 Chi tiết": "detailed"}

# ── Confidence thresholds (heuristic) ────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "high": {"min_sources": 4, "label": "Cao", "color": "#27ae60", "icon": "🟢"},
    "medium": {"min_sources": 2, "label": "Trung bình", "color": "#f39c12", "icon": "🟡"},
    "low": {"min_sources": 0, "label": "Thấp", "color": "#e74c3c", "icon": "🔴"},
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "rag_query.log"
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT = 3

# ── Pipeline description ──────────────────────────────────────────────────────
PIPELINE_DESC = (
    "PDF → Recursive-Char Chunking → BGE-M3 Embed → BM25 / Dense MMR Retrieval "
    "→ (BGE Reranker nếu Quality) → GPT-4o-mini"
)
