"""
ingest_political_6.py — Ingest 6 giáo trình chính trị với pipeline:
sentence-aware chunking + intfloat/multilingual-e5-large + BM25.

Chạy:
    python ingest_political_6.py
    python ingest_political_6.py --subjects "Lịch sử Đảng Cộng sản Việt Nam" "Tư tưởng Hồ Chí Minh"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ingest import ingest

SUBJECT_INGEST_MAP: dict[str, dict[str, str]] = {
    "Lịch sử Đảng Cộng sản Việt Nam": {
        "pdf": "data/lich-su-dang-cong-san-viet-nam-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_lich_su_dang_e5_bm25",
        "collection": "rag_lich_su_dang_e5_bm25",
    },
    "Pháp luật đại cương": {
        "pdf": "data/phap-luat-dai-cuong-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_phap_luat_dai_cuong_e5_bm25",
        "collection": "rag_pldc_e5_bm25",
    },
    "Triết học Mác-Lênin": {
        "pdf": "data/triet-hoc-mac-lenin-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_triet_hoc_e5_bm25",
        "collection": "rag_triet_hoc_e5_bm25",
    },
    "Kinh tế Chính trị Mác-Lênin": {
        "pdf": "data/kinh-te-chinh-tri-mac-lenin-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_kinh_te_chinh_tri_e5_bm25",
        "collection": "rag_kinh_te_chinh_tri_e5_bm25",
    },
    "Chủ nghĩa Xã hội Khoa học": {
        "pdf": "data/chu-nghia-xa-hoi-khoa-hoc-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_chu_nghia_xa_hoi_khoa_hoc_e5_bm25",
        "collection": "rag_chu_nghia_xa_hoi_khoa_hoc_e5_bm25",
    },
    "Tư tưởng Hồ Chí Minh": {
        "pdf": "data/tu-tuong-hcm-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_tu_tuong_hcm_e5_bm25",
        "collection": "rag_tu_tuong_hcm_e5_bm25",
    },
}


def run_ingest(subjects: list[str], force_rebuild: bool) -> None:
    model_name = "intfloat/multilingual-e5-large"
    for idx, subject in enumerate(subjects, 1):
        cfg = SUBJECT_INGEST_MAP[subject]
        pdf_path = Path(cfg["pdf"])
        chroma_dir = Path(cfg["chroma_dir"])
        collection = cfg["collection"]

        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")

        if force_rebuild and chroma_dir.exists():
            print(f"[{idx}/{len(subjects)}] Xóa DB cũ: {chroma_dir}")
            shutil.rmtree(chroma_dir)

        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(subjects)}] Ingest môn: {subject}")
        print(f"PDF       : {pdf_path}")
        print(f"Chroma dir: {chroma_dir}")
        print(f"Collection: {collection}")
        print(f"Embedding : {model_name}")
        print(f"{'=' * 70}")

        ingest(
            pdf_path=str(pdf_path),
            chroma_dir=str(chroma_dir),
            collection_name=collection,
            embedding_model=model_name,
            chunk_max_chars=1000,
            chunk_overlap_sents=2,
            batch_size=100,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest 6 giáo trình chính trị với E5-Large + BM25"
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        choices=list(SUBJECT_INGEST_MAP.keys()),
        help="Danh sách môn muốn ingest (mặc định: ingest toàn bộ 6 môn)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Xóa thư mục Chroma cũ trước khi ingest",
    )
    args = parser.parse_args()

    selected_subjects = args.subjects or list(SUBJECT_INGEST_MAP.keys())
    run_ingest(selected_subjects, args.force_rebuild)
