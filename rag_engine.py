"""
rag_engine.py — Class HistoryRAG:
Recursive-char chunks + HuggingFace embeddings (BGE-M3, VN bi-encoder FT, …)
+ BM25 / Dense-MMR retrieval + optional CrossEncoder reranking + LLM generation.

Hỗ trợ 3 pipeline mode:
  fast    → BM25, top_k=3, no rerank
  balance → BM25, top_k=5, no rerank
  quality → Dense MMR, top_k=5, CrossEncoder reranker (BAAI/bge-reranker-v2-m3)
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_VI_BI_ENCODER_DIR = _REPO_ROOT / "models" / "vietnamese-bi-encoder-finetuned"
_DOWNLOADED_VI_BI_ENCODER_DIR = (
    _REPO_ROOT
    / "bi-encoder-finetuned"
    / "models"
    / "bi_encoder_hnm_v2"
    / "vietnamese-bi-encoder-v2-hnm"
)
_DEFAULT_VI_BI_ENCODER = str(
    _DEFAULT_VI_BI_ENCODER_DIR
    if _DEFAULT_VI_BI_ENCODER_DIR.exists()
    else _DOWNLOADED_VI_BI_ENCODER_DIR
)

# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """Bạn là một chuyên gia về lịch sử và các môn lý luận chính trị Việt Nam,
có kiến thức sâu rộng về nội dung giáo trình đại học được cung cấp.

NGUYÊN TẮC TRẢ LỜI BẮT BUỘC:
1. Chỉ trả lời dựa trên thông tin có trong NGỮ CẢNH được cung cấp bên dưới.
2. Nếu câu hỏi không có thông tin trong ngữ cảnh, hãy trả lời:
   "Tôi không tìm thấy thông tin này trong giáo trình được cung cấp."
3. TUYỆT ĐỐI KHÔNG tự ý bịa đặt thông tin.
4. Trích dẫn rõ nguồn (số trang nếu có).
5. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc.
6. Nếu thiếu thông tin, phải nói rõ.

NGỮ CẢNH:
{context}
"""

_SUPPORTED_EMBEDDING_MODELS = {
    "E5-Large (Local)": "intfloat/multilingual-e5-large",
    "BGE-M3 (Local)": "BAAI/bge-m3",
    # Fine-tuned từ notebook: đặt model tại models/vietnamese-bi-encoder-finetuned
    # hoặc gán UNIPOLIS_VI_BI_ENCODER_PATH (đường dẫn tuyệt đối tới thư mục đã save).
    "Vietnamese Bi-Encoder (FT)": os.environ.get(
        "UNIPOLIS_VI_BI_ENCODER_PATH", _DEFAULT_VI_BI_ENCODER
    ),
    "Gemini": "google",
    "Google": "google",
}


class HistoryRAG:
    """
    Encapsulates the RAG pipeline.
    retriever_mode: 'bm25' | 'dense' | 'dense_mmr' | 'hybrid'
    use_rerank: nếu True → dùng CrossEncoder (reranker_model) sau retrieval.
    """

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str = "rag_collection",
        embedding_model_key: str = "BGE-M3 (Local)",
        llm_model: str = "gemini-3.1-flash-lite-preview",
        rerank_top_k: int = 5,
        retriever_mode: str = "bm25",
        dense_fetch_k: int = 25,
        use_rerank: bool = False,
        reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.top_k = rerank_top_k
        self.retriever_mode = retriever_mode
        self.dense_fetch_k = dense_fetch_k
        self.use_rerank = use_rerank
        self._reranker_model_name = reranker_model
        self._reranker = None  # lazy-loaded CrossEncoder

        # ── Embeddings ──────────────────────────────────────────────────────
        self.embeddings = self._load_embeddings(embedding_model_key)

        # ── Chroma ──────────────────────────────────────────────────────────
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )

        # ── BM25 ────────────────────────────────────────────────────────────
        bm25_path = Path(chroma_dir) / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index không tìm thấy tại {bm25_path}. "
                "Hãy chạy ingest.py trước."
            )
        with open(bm25_path, "rb") as f:
            self.bm25_retriever: BM25Retriever = pickle.load(f)
        self.bm25_retriever.k = self.top_k

        # ── LLM chain ───────────────────────────────────────────────────────
        self.llm_model = llm_model
        self._chain = None

    def _get_chain(self):
        """Lazy-load Gemini only when generation is actually requested."""
        if self._chain is not None:
            return self._chain

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY chưa được cấu hình. Retrieval vẫn dùng được, "
                "nhưng cần API key để sinh câu trả lời."
            )

        llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=0.1,
            max_tokens=2048,
            api_key=api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", "{question}")]
        )
        self._chain = prompt | llm | StrOutputParser()
        return self._chain

    # ── Public API ────────────────────────────────────────────────────────────
    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Return (answer_text, source_docs)."""
        docs = self._retrieve(question)
        context = self._format_context(docs)
        answer = self._get_chain().invoke({"context": context, "question": question})
        return answer, docs

    # ── Private helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _load_embeddings(model_key: str):
        model_name = _SUPPORTED_EMBEDDING_MODELS.get(model_key, "BAAI/bge-m3")

        if model_name == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )

    def _get_reranker(self):
        """Lazy-load CrossEncoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                self._reranker_model_name,
                max_length=512,
            )
        return self._reranker

    def _rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank docs với CrossEncoder, trả về danh sách đã sắp xếp theo score giảm dần."""
        reranker = self._get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]

    def _retrieve(self, query: str) -> List[Document]:
        mode = self.retriever_mode

        if mode == "dense":
            docs = self.vectorstore.similarity_search(query, k=self.top_k)

        elif mode == "dense_mmr":
            # Max Marginal Relevance — đa dạng hơn pure similarity search
            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.top_k,
                fetch_k=self.dense_fetch_k,
            )

        elif mode == "hybrid":
            dense_docs = self.vectorstore.similarity_search(query, k=self.dense_fetch_k)
            sparse_docs = self.bm25_retriever.invoke(query)
            merged: List[Document] = []
            seen: set = set()
            for doc in sparse_docs + dense_docs:
                key = doc.metadata.get("chunk_uid") or str(hash(doc.page_content))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
                if len(merged) >= self.top_k:
                    break
            docs = merged

        else:  # bm25 (default)
            docs = self.bm25_retriever.invoke(query)

        # Optional CrossEncoder reranking (Quality mode)
        if self.use_rerank and docs:
            docs = self._rerank_docs(query, docs)[: self.top_k]

        return docs

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            try:
                page_display = int(page) + 1
            except (ValueError, TypeError):
                page_display = page
            parts.append(f"[Đoạn {i} - Trang {page_display}]\n{doc.page_content}")
        sep = "\n\n" + "=" * 40 + "\n\n"
        return sep.join(parts)
