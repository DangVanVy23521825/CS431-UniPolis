# Demo pipeline mới: sentence-aware + E5-large + BM25 + no-rerank

## 1) Cài dependencies

```bash
pip install -r requirements.txt
```

## 2) Ingest 6 PDF chính trị

```bash
python ingest_political_6.py --force-rebuild
```

Nếu muốn ingest riêng một vài môn:

```bash
python ingest_political_6.py --subjects "Lịch sử Đảng Cộng sản Việt Nam" "Pháp luật đại cương"
```

## 3) Chạy demo app

```bash
streamlit run app.py
```

Trong app:
- Chọn môn học ở sidebar.
- Embedding mặc định: `E5-Large (Local)`.
- Top-K: số đoạn BM25 trả về cho câu hỏi.

## 4) Pipeline đang dùng

`PDF -> sentence-aware chunking -> intfloat/multilingual-e5-large -> Chroma + BM25 index -> BM25 retrieval (no rerank) -> GPT-4o-mini`
