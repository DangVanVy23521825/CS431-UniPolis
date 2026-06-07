# UniPolis

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CS431](https://img.shields.io/badge/Môn-CS431-8e44ad?style=flat-square)](docs/DAC_TA_DO_AN_CS431.md)
[![Sentence Transformers](https://img.shields.io/badge/Embedding-Vietnamese_Bi--Encoder-27ae60?style=flat-square)](https://www.sbert.net/)

**Fine-tune Vietnamese Bi-Encoder V2, benchmark retrieval và đánh giá RAGAS** cho hệ thống RAG giáo trình lý luận chính trị tiếng Việt (đồ án CS431).

[Tổng quan](#tổng-quan) • [Cấu trúc repo](#cấu-trúc-repo) • [Bắt đầu](#bắt-đầu) • [Đánh giá](#đánh-giá) • [Notebook](#notebook) • [Demo](#demo) • [FAQ](#faq)

---

## Tổng quan

Repo này tập trung vào **đóng góp học sâu** của đồ án UniPolis:

- **Fine-tune Vietnamese Bi-Encoder V2** với hard negative mining trên QA pairs tự sinh từ giáo trình
- **Split train/val/test** theo `(môn, chapter)` — tránh data leakage
- **Benchmark retrieval** — MRR@10, Recall@k (base, BGE-M3, E5-Large, V2)
- **Đánh giá downstream RAGAS** — Faithfulness, Context Precision/Recall, Answer Relevancy

> [!NOTE]
> **Ứng dụng demo Streamlit** do teammate phụ trách — không nằm trong repo này. Xem [demo/README.md](demo/README.md) để tích hợp model V2.

### Kết quả chính

| Metric | Base VN Bi-Encoder | V2 hard negative | Δ |
|--------|-------------------|----------------|---|
| MRR@10 (test, 611 queries) | 0.760 | 0.844 | +8.4% |
| MRR@10 (notebook val/test) | 0.418 | 0.498 | +8.0% |
| Context Recall (RAGAS, 30 PLDC) | 0.902 | 0.993 | +9.2% |

---

## Cấu trúc repo

```text
cs431/
├── Notebook/                 # Fine-tune, sinh/lọc data, RAGAS Kaggle
├── scripts/                  # Benchmark retrieval & RAGAS
├── data/
│   ├── *.pdf                 # 6 giáo trình
│   └── training_data/        # qa_audit.jsonl (3 môn)
├── models/                   # Metadata huấn luyện (weights trên Hugging Face)
├── artifacts/                # Kết quả MRR, RAGAS
├── docs/                     # Đặc tả & báo cáo
└── demo/                     # ← Teammate push ứng dụng demo vào đây
```

---

## Bắt đầu

### Yêu cầu

- Python 3.10+
- GPU khuyến nghị cho fine-tune (Kaggle T4 đủ dùng)
- `GOOGLE_API_KEY` — chỉ cần khi chạy script RAGAS

### Cài đặt

```bash
git clone https://github.com/DangVanVy23521825/CS431-UniPolis.git
cd CS431-UniPolis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Model V2 (Hugging Face)

Weights (~517 MB) **không** nằm trong Git. Model V2 được host trên Hugging Face — chỉ cần gắn biến môi trường; lần chạy đầu sẽ tự tải về cache (`~/.cache/huggingface/hub/`).

1. Copy `.env.example` → `.env` (nếu chưa có)
2. Thêm vào `.env`:

```bash
# Repo public — không cần login Hugging Face
UNIPOLIS_VI_BI_ENCODER_PATH=HoangNgo1026/vietnamese-bi-encoder-v2-hnm
```

3. (Tuỳ chọn) Nếu chạy demo / RAGAS cần Gemini:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

Repo **private** hoặc **gated** → thêm `HF_TOKEN=hf_...` hoặc chạy `hf auth login`.

Model card: [HoangNgo1026/vietnamese-bi-encoder-v2-hnm](https://huggingface.co/HoangNgo1026/vietnamese-bi-encoder-v2-hnm) · Chi tiết metadata: [models/README.md](models/README.md)

---

## Đánh giá

### Retrieval — `scripts/compare_biencoder_v2_retrieval.py`

```bash
python scripts/compare_biencoder_v2_retrieval.py --training-base data/training_data
```

Kết quả: `artifacts/compare_biencoder_v2_retrieval.csv`

| Model | MRR@10 | Recall@5 |
|-------|--------|----------|
| Base VN Bi-Encoder | 0.760 | 0.905 |
| BGE-M3 | 0.873 | 0.967 |
| E5-Large | 0.866 | 0.957 |
| **V2 hard negative** | **0.844** | **0.953** |

### RAGAS downstream — `scripts/run_ragas_retriever_comparison.py`

```bash
python scripts/run_ragas_retriever_comparison.py \
  --training-base data/training_data \
  --source training_data_pldc \
  --sample-size 30
```

Kết quả: `artifacts/ragas_retriever_comparison_pldc_30_retry/ragas_summary_full.csv`

| Model | Faithfulness | Answer Relevancy* | Context Precision | Context Recall |
|-------|--------------|-------------------|-------------------|----------------|
| Base | 0.960 | 0.950 | 0.784 | 0.902 |
| **V2** | 0.944 | **0.983** | **0.826** | **0.993** |

\*Answer Relevancy chấm thủ công (`answer_relevancy_source=manual`).

---

## Notebook

| Notebook | Vai trò |
|----------|---------|
| [`fine-tuning-vietnamese-bi-encoder-v2.ipynb`](Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb) | **Chính** — train V1/V2, HNM, eval |
| [`data_generation.ipynb`](Notebook/data_generation.ipynb) | Sinh QA từ PDF |
| [`data_filter.ipynb`](Notebook/data_filter.ipynb) | Lọc QA kém |
| [`compare.ipynb`](Notebook/compare.ipynb) | So sánh embedding đa môn |
| [`ragas-kaggle-local.ipynb`](Notebook/ragas-kaggle-local.ipynb) | RAGAS E2E với Qwen (Kaggle) |
| [`fine-tuning-bge-m3-v2.ipynb`](Notebook/fine-tuning-bge-m3-v2.ipynb) | Baseline BGE-M3 |

> [!TIP]
> Fine-tune trên Kaggle: add dataset `dangvy1507/training-data`, bật GPU + Internet.

---

## Demo

Ứng dụng demo (Streamlit, ingest, RAG engine) **không** nằm trong repo này — teammate sẽ push riêng.

Để tích hợp model V2 vào demo:

1. Set `UNIPOLIS_VI_BI_ENCODER_PATH` trong `.env` (xem [Bắt đầu](#model-v2-hugging-face))
2. Ingest Chroma/BM25 bằng **cùng model V2** và chunking thống nhất
3. Chọn pipeline dense dùng `Vietnamese Bi-Encoder (FT)`

---

## Tài liệu

| File | Nội dung |
|------|----------|
| [docs/DAC_TA_DO_AN_CS431.md](docs/DAC_TA_DO_AN_CS431.md) | Đặc tả đồ án |
| [docs/BAO_CAO_UNIPOLIS_NOI_DUNG_HOC_THUAT.md](docs/BAO_CAO_UNIPOLIS_NOI_DUNG_HOC_THUAT.md) | Nội dung học thuật |
| [docs/huong-dan-fine-tuning-bi-encoder-rag.md](docs/huong-dan-fine-tuning-bi-encoder-rag.md) | Hướng dẫn fine-tune |

---

## FAQ

**Repo có chứa model weights không?**  
Không. Set `UNIPOLIS_VI_BI_ENCODER_PATH=HoangNgo1026/vietnamese-bi-encoder-v2-hnm` trong `.env` — model tự tải từ Hugging Face (~517 MB, cần Internet lần đầu).

**Chunking lúc fine-tune khác lúc demo thì sao?**  
Model vẫn chạy nhưng retrieval có thể kém do distribution shift. Nên đồng bộ chunk giữa sinh data → fine-tune → ingest demo.

**Ai phụ trách phần nào?**  
Fine-tune, benchmark, RAGAS → repo này. Demo UI → teammate (thư mục `demo/`).

---

## Tài nguyên

- [Vietnamese Bi-Encoder (base)](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder)
- [**V2 hard negative (UniPolis)**](https://huggingface.co/HoangNgo1026/vietnamese-bi-encoder-v2-hnm)
- [RAGAS](https://docs.ragas.io/) · [Sentence Transformers](https://www.sbert.net/)
- Kaggle (training data): `dangvy1507/training-data`

---

*CS431 — UniPolis: Fine-tune bi-encoder & đánh giá RAG cho giáo trình lý luận chính trị.*
