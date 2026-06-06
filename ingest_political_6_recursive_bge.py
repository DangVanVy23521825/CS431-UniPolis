"""
ingest_political_6_recursive_bge.py — Ingest 6 giáo trình chính trị theo cấu hình:
recursive_char chunking + BAAI/bge-m3 + BM25.

Tham chiếu cấu hình từ notebook `rag (2).ipynb`:
- chunk_size=512
- chunk_overlap=64
- separators=["\\n\\n", "\\n", ".", " ", ""]

Chạy:
    python ingest_political_6_recursive_bge.py --force-rebuild
    python ingest_political_6_recursive_bge.py --subjects "Lịch sử Đảng Cộng sản Việt Nam"
    python ingest_political_6_recursive_bge.py --force-rebuild --profile vn-bi-ft --model models/vietnamese-bi-encoder-finetuned
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import build_embeddings, clean_page_text, load_pdf

SUBJECT_INGEST_MAP: dict[str, dict[str, str]] = {
    "Lịch sử Đảng Cộng sản Việt Nam": {
        "pdf": "data/lich-su-dang-cong-san-viet-nam-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_lich_su_dang_recursive_bge_bm25",
        "collection": "rag_lich_su_dang_recursive_bge_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_lich_su_dang_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_lich_su_dang_recursive_vn_bi_ft_bm25",
    },
    "Pháp luật đại cương": {
        "pdf": "data/phap-luat-dai-cuong-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_phap_luat_recursive_bge_bm25",
        "collection": "rag_phap_luat_recursive_bge_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_phap_luat_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_phap_luat_recursive_vn_bi_ft_bm25",
    },
    "Triết học Mác-Lênin": {
        "pdf": "data/triet-hoc-mac-lenin-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_triet_hoc_recursive_bge_bm25",
        "collection": "rag_triet_hoc_recursive_bge_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_triet_hoc_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_triet_hoc_recursive_vn_bi_ft_bm25",
    },
    "Kinh tế Chính trị Mác-Lênin": {
        "pdf": "data/kinh-te-chinh-tri-mac-lenin-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_kinh_te_recursive_bge_bm25",
        "collection": "rag_kinh_te_recursive_bge_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_kinh_te_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_kinh_te_recursive_vn_bi_ft_bm25",
    },
    "Chủ nghĩa Xã hội Khoa học": {
        "pdf": "data/chu-nghia-xa-hoi-khoa-hoc-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_cnxhkh_recursive_bge_bm25",
        "collection": "rag_cnxhkh_recursive_bge_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_cnxhkh_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_cnxhkh_recursive_vn_bi_ft_bm25",
    },
    "Tư tưởng Hồ Chí Minh": {
        "pdf": "data/tu-tuong-hcm-giao-trinh.pdf",
        "chroma_dir": "data/chroma_db_tthcm_recursive_bge_bm25",
        "collection": "rag_tthcm_recursive_bge_bm25",
        "chroma_dir_vn_bi_ft": "data/chroma_db_tthcm_recursive_vn_bi_ft_bm25",
        "collection_vn_bi_ft": "rag_tthcm_recursive_vn_bi_ft_bm25",
    },
}

_LLM_GARBAGE = [
    "Vui lòng cung cấp nội dung",
    "Tôi đã sẵn sàng",
    "Dưới đây là nội dung đã được làm sạch",
]


def recursive_char_chunk(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks, 1):
        chunk.metadata["chunk_uid"] = f"rc{i}"
        chunk.metadata["chunk_method"] = "recursive_char"
    return chunks


def ingest_one_subject(
    pdf_path: Path,
    chroma_dir: Path,
    collection_name: str,
    batch_size: int = 100,
    embedding_model: str = "BAAI/bge-m3",
) -> None:
    print(f"[1/5] Đọc PDF: {pdf_path}")
    pages = load_pdf(str(pdf_path))
    print(f"      → {len(pages)} trang")

    print("[2/5] Làm sạch text...")
    cleaned = [
        Document(page_content=clean_page_text(d.page_content), metadata=d.metadata)
        for d in pages
    ]
    cleaned = [
        d
        for d in cleaned
        if d.page_content and len(d.page_content) > 50
        and not any(g in d.page_content for g in _LLM_GARBAGE)
    ]
    print(f"      → {len(cleaned)} trang còn lại sau lọc")

    print("[3/5] Chunking (recursive_char)...")
    chunks = recursive_char_chunk(cleaned, chunk_size=512, chunk_overlap=64)
    print(f"      → {len(chunks)} chunks")

    print(f"[4/5] Tạo Chroma ({collection_name}) tại {chroma_dir}...")
    embeddings = build_embeddings(embedding_model)
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(chroma_dir),
            )
        else:
            vectorstore.add_documents(batch)
        print(f"      Chroma: {min(i + batch_size, len(chunks))}/{len(chunks)}")
    print(f"      → {vectorstore._collection.count()} vectors đã lưu")

    print("[5/5] Lưu BM25 index...")
    bm25_path = chroma_dir / "bm25_index.pkl"
    bm25 = BM25Retriever.from_documents(chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"      → BM25 index lưu tại {bm25_path}")
    print("✅ Ingest hoàn tất!")


def run_ingest(
    subjects: list[str], force_rebuild: bool, embedding_model: str, profile: str
) -> None:
    for idx, subject in enumerate(subjects, 1):
        cfg = SUBJECT_INGEST_MAP[subject]
        pdf_path = Path(cfg["pdf"])
        if profile == "vn-bi-ft":
            chroma_dir = Path(cfg["chroma_dir_vn_bi_ft"])
            collection = cfg["collection_vn_bi_ft"]
        else:
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
        print(f"Embedding : {embedding_model}")
        print(f"Profile   : {profile}")
        print("Chunking  : recursive_char (size=512, overlap=64)")
        print(f"{'=' * 70}")

        ingest_one_subject(
            pdf_path=pdf_path,
            chroma_dir=chroma_dir,
            collection_name=collection,
            batch_size=100,
            embedding_model=embedding_model,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest 6 giáo trình chính trị với recursive chunk + embedding HF + BM25"
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
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Embedding HuggingFace: repo id hoặc đường dẫn thư mục model đã save "
        "(vd. models/vietnamese-bi-encoder-finetuned)",
    )
    parser.add_argument(
        "--profile",
        choices=["bge", "vn-bi-ft"],
        default="bge",
        help="Nhóm đường dẫn Chroma/collection cần build. Dùng vn-bi-ft khi ingest bằng Bi-Encoder fine-tuned.",
    )
    args = parser.parse_args()

    selected_subjects = args.subjects or list(SUBJECT_INGEST_MAP.keys())
    run_ingest(selected_subjects, args.force_rebuild, args.model, args.profile)
