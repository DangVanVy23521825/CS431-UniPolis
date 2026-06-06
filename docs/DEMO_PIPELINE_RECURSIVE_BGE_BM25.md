# Demo pipeline: recursive chunk + BGE-M3 + BM25

Pipeline này bám theo cấu hình trong `rag (2).ipynb`:
- Chunking: `recursive_char`
- `chunk_size=512`
- `chunk_overlap=64`
- `separators=["\n\n", "\n", ".", " ", ""]`
- Embedding: `BAAI/bge-m3`
- Retrieval: `BM25`
- Reranking: `None`
- LLM: `gpt-4o-mini`

## Chạy ingest cho 6 PDF

```bash
python ingest_political_6_recursive_bge.py --force-rebuild
```

## Chạy ingest cho một vài môn

```bash
python ingest_political_6_recursive_bge.py --subjects "Lịch sử Đảng Cộng sản Việt Nam" "Tư tưởng Hồ Chí Minh"
```
