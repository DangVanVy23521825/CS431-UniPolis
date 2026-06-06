"""
ingest.py — Tạo Chroma vector store + BM25 index từ file PDF.

Cách dùng:
    python ingest.py --pdf data/lich-su-dang.pdf \
                     --chroma_dir data/chroma_db_lich_su_dang \
                     --collection rag_configB_bge_m3
"""

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from underthesea import sent_tokenize

load_dotenv()

# ── Regex làm sạch ──────────────────────────────────────────────────────────
_STUDOCU_RE = re.compile(
    r"Scan to open on Studocu\s*"
    r"|Studocu is not sponsored or endorsed[^\n]*\n?"
    r"|Lịch sử Đảng Cộng Sản Việt Nam - Giáo Trình\s*\n?"
    r"|Lịch Sử Đảng Cộng Sản Việt Nam \(Học viện[^)]*\)\s*\n?",
    re.IGNORECASE,
)
_FOOTER_RE = re.compile(
    r"Downloaded by [^\n]+\n?"
    r"lOMoARcPSD\|[^\n]*\n?",
    re.IGNORECASE,
)
_PAGE_NUM_HEADER_RE = re.compile(r"^ \n\d{1,3} \n \n")

_LLM_GARBAGE = [
    "Vui lòng cung cấp nội dung",
    "Tôi đã sẵn sàng",
    "Dưới đây là nội dung đã được làm sạch",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_pdf(file_path: str) -> List[Document]:
    doc = fitz.open(file_path)
    return [
        Document(
            page_content=doc[i].get_text(),
            metadata={"source": file_path, "page": i},
        )
        for i in range(len(doc))
    ]


def clean_page_text(text: str) -> str:
    text = _STUDOCU_RE.sub("", text)
    text = _FOOTER_RE.sub("", text)
    text = _PAGE_NUM_HEADER_RE.sub("", text, count=1)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_aware_chunk(
    documents: List[Document],
    max_chunk_chars: int = 1000,
    overlap_sentences: int = 2,
) -> List[Document]:
    chunks: List[Document] = []
    uid_n = 0
    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        sents = sent_tokenize(text) or [text]
        i = 0
        while i < len(sents):
            group: List[str] = []
            total = 0
            j = i
            while j < len(sents):
                s = sents[j]
                add = len(s) + (1 if group else 0)
                if group and total + add > max_chunk_chars:
                    break
                if not group and len(s) > max_chunk_chars:
                    group.append(s[:max_chunk_chars])
                    total = len(group[0])
                    j += 1
                    break
                group.append(s)
                total += add
                j += 1
            if not group:
                break
            uid_n += 1
            meta = {**doc.metadata, "chunk_uid": f"c{uid_n}"}
            chunks.append(Document(page_content=" ".join(group).strip(), metadata=meta))
            if j >= len(sents):
                break
            next_i = max(i + 1, j - overlap_sentences)
            if next_i >= len(sents):
                break
            i = next_i
    return chunks


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def ingest(
    pdf_path: str,
    chroma_dir: str,
    collection_name: str,
    embedding_model: str = "BAAI/bge-m3",
    chunk_max_chars: int = 1000,
    chunk_overlap_sents: int = 2,
    batch_size: int = 100,
) -> None:
    print(f"[1/5] Đọc PDF: {pdf_path}")
    pages = load_pdf(pdf_path)
    print(f"      → {len(pages)} trang")

    print("[2/5] Làm sạch text...")
    cleaned = [
        Document(page_content=clean_page_text(d.page_content), metadata=d.metadata)
        for d in pages
    ]
    cleaned = [
        d for d in cleaned
        if d.page_content and len(d.page_content) > 50
        and not any(g in d.page_content for g in _LLM_GARBAGE)
    ]
    print(f"      → {len(cleaned)} trang còn lại sau lọc")

    print("[3/5] Chunking (sentence-aware)...")
    chunks = sentence_aware_chunk(cleaned, chunk_max_chars, chunk_overlap_sents)
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
                persist_directory=chroma_dir,
            )
        else:
            vectorstore.add_documents(batch)
        print(f"      Chroma: {min(i + batch_size, len(chunks))}/{len(chunks)}")
    print(f"      → {vectorstore._collection.count()} vectors đã lưu")

    print("[5/5] Lưu BM25 index...")
    bm25_path = Path(chroma_dir) / "bm25_index.pkl"
    bm25 = BM25Retriever.from_documents(chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"      → BM25 index lưu tại {bm25_path}")
    print("\n✅ Ingest hoàn tất!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo Chroma DB + BM25 từ PDF")
    parser.add_argument("--pdf", required=True, help="Đường dẫn tới file PDF")
    parser.add_argument("--chroma_dir", required=True, help="Thư mục lưu Chroma")
    parser.add_argument("--collection", default="rag_collection", help="Tên collection")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Tên embedding model")
    parser.add_argument("--max_chars", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=2)
    args = parser.parse_args()

    ingest(
        pdf_path=args.pdf,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.model,
        chunk_max_chars=args.max_chars,
        chunk_overlap_sents=args.overlap,
    )
