# Demo application

Streamlit RAG demo — VN Bi-Encoder V2 · Dense MMR · Gemini.

## Chạy demo

Từ **thư mục gốc** repo (cần `.env`, `data/`, `models/`):

```bash
streamlit run demo/app.py
```

## Chuẩn bị

1. `GOOGLE_API_KEY` trong `.env` ở thư mục gốc
2. Model V2 tại `models/vietnamese-bi-encoder-v2-hnm/`
3. Vector store đã ingest (`data/chroma_db_*_vn_bi_ft_bm25`) cho 3 môn hỗ trợ

Chi tiết: [models/README.md](../models/README.md)
