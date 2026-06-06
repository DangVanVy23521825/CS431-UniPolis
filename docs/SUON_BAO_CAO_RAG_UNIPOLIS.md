# Sườn báo cáo đồ án CS431 - UniPolis theo hướng RAG

Tài liệu này được thiết kế lại dựa trên các báo cáo mẫu liên quan đến RAG/hỏi đáp học tập trong môn CS431, đặc biệt là cấu trúc của các báo cáo TiepLM và hệ thống hỏi đáp từ video bài giảng. Sườn này phù hợp hơn cho UniPolis so với kiểu báo cáo paper tối ưu hóa thuần túy, vì đề tài hiện tại là một hệ thống RAG có thực nghiệm retrieval, fine-tuning và generation.

## Cấu trúc tổng thể đề xuất

```text
Trang bìa
Mục lục
1. Tóm tắt
2. Giới thiệu đề tài
3. Phương pháp thực hiện
4. Dữ liệu và độ đo đánh giá
5. Kết quả đánh giá
6. Những cải tiến và đóng góp học sâu
7. Kết luận và hướng phát triển
8. Bảng phân công
Tài liệu tham khảo
Phụ lục
```

## Trang bìa

Nội dung nên có:

- Đại học Quốc gia TP.HCM.
- Trường Đại học Công nghệ Thông tin.
- Khoa/ngành nếu cần.
- Môn học: Các kỹ thuật học sâu và ứng dụng - CS431.
- Tên đề tài.
- Giảng viên hướng dẫn.
- Sinh viên thực hiện.
- Thành phố Hồ Chí Minh, năm thực hiện.

Tên đề tài đề xuất:

**UniPolis - Hệ thống RAG hỗ trợ hỏi đáp và sinh câu hỏi từ giáo trình lý luận chính trị tiếng Việt**

Nếu muốn nhấn mạnh học sâu hơn:

**UniPolis - Hệ thống RAG tiếng Việt kết hợp fine-tuning Bi-Encoder và QLoRA LLM nhỏ cho giáo trình lý luận chính trị**

## 1. Tóm tắt

Viết 1 trang hoặc khoảng 250-350 từ.

Nội dung nên có:

- Bối cảnh: sinh viên phải học nhiều giáo trình lý luận chính trị dài, nội dung trừu tượng, khó tra cứu nhanh.
- Vấn đề: tìm kiếm PDF thủ công kém hiệu quả; LLM thuần dễ hallucination và không có nguồn kiểm chứng.
- Giải pháp: xây dựng UniPolis, một trợ lý học tập dựa trên Retrieval-Augmented Generation.
- Thành phần chính: xử lý PDF, chunking, ChromaDB, BM25, dense retrieval, reranking, LLM generation, giao diện Streamlit.
- Đóng góp học sâu: fine-tuning Vietnamese Bi-Encoder V2 bằng MultipleNegativesRankingLoss và hard negative mining; fine-tuning Qwen2.5-1.5B-Instruct bằng QLoRA trên instruction data giáo trình.
- Kết quả chính:
  - Bi-Encoder V2 đạt `MRR@10 = 0.4982` trên test set độc lập.
  - QLoRA Qwen đạt `test_loss = 0.0807`, `JSON valid rate = 1.0` trên quick generation eval.
  - RAG app hỗ trợ hỏi đáp, nguồn tham chiếu và tạo đề/câu hỏi.

Gợi ý đoạn kết tóm tắt:

> Kết quả cho thấy việc kết hợp RAG với fine-tuning mô hình retrieval và instruction-tuning LLM nhỏ giúp cải thiện khả năng truy hồi, kiểm soát nguồn và sinh nội dung học tập có cấu trúc trong miền tiếng Việt học thuật.

## 2. Giới thiệu đề tài

### 2.1. Bối cảnh và vấn đề

Viết theo hướng giống báo cáo TiepLM:

- Sinh viên cần tra cứu, ôn tập và tự kiểm tra kiến thức từ nhiều giáo trình dài.
- Giáo trình lý luận chính trị có nhiều khái niệm, phạm trù, luận điểm và mốc lịch sử.
- Các công cụ tìm kiếm từ khóa trong PDF không xử lý tốt câu hỏi tự nhiên.
- LLM trả lời tốt về mặt ngôn ngữ nhưng thiếu đảm bảo đúng giáo trình nếu không có cơ chế truy hồi nguồn.

Điểm nhấn:

> UniPolis giải quyết bài toán hỏi đáp có căn cứ trên tài liệu học tập, trong đó câu trả lời không chỉ cần đúng ngữ nghĩa mà còn cần có nguồn kiểm chứng.

### 2.2. Mục tiêu và giải pháp

Mục tiêu tổng quát:

- Xây dựng trợ lý học tập tiếng Việt cho 6 giáo trình lý luận chính trị dựa trên RAG.

Mục tiêu cụ thể:

- Xử lý PDF thành các đoạn văn bản có thể truy hồi.
- Xây dựng chỉ mục vector và BM25 theo từng môn.
- Cho phép người dùng hỏi đáp tự nhiên và xem nguồn tham chiếu.
- Hỗ trợ tạo đề/câu hỏi ôn tập.
- Đánh giá các chiến lược retrieval bằng Recall/MRR.
- Fine-tune Vietnamese Bi-Encoder để cải thiện dense retrieval.
- Fine-tune Qwen2.5-1.5B-Instruct bằng QLoRA để sinh JSON học tập có cấu trúc.

### 2.3. Các chức năng chính

| Chức năng | Mô tả |
|-----------|-------|
| Hỏi đáp theo giáo trình | Người dùng chọn môn, đặt câu hỏi và nhận câu trả lời dựa trên ngữ cảnh truy hồi. |
| Hiển thị nguồn tham chiếu | Câu trả lời đi kèm các đoạn/chunk/trang liên quan để kiểm chứng. |
| Chọn pipeline retrieval | Cho phép thay đổi chế độ fast/balance/quality hoặc model embedding. |
| Tạo đề/câu hỏi | Sinh câu hỏi tự luận/trắc nghiệm dựa trên nội dung giáo trình. |
| So sánh retrieval | Đánh giá nhiều embedding/pipeline bằng Recall@k, MRR@k và latency. |
| Fine-tuning retrieval | Huấn luyện Vietnamese Bi-Encoder V2 với hard negative mining. |
| Fine-tuning generation | QLoRA Qwen để sinh `grounded_qa`, `question_generation`, `mcq_generation`, `summary` theo JSON. |

### 2.4. Đóng góp chính

Nên liệt kê 4-6 đóng góp:

- Một pipeline RAG end-to-end cho 6 giáo trình lý luận chính trị tiếng Việt.
- Thiết kế dữ liệu theo từng môn để giảm nhiễu liên miền.
- Thực nghiệm so sánh embedding/retrieval với metric định lượng.
- Fine-tuning Vietnamese Bi-Encoder V2 có test set độc lập theo `(source, chapter)`.
- Hard negative mining giúp cải thiện test MRR@10 từ `0.4795` lên `0.4982` so với V1.
- QLoRA Qwen2.5-1.5B-Instruct cho generation có cấu trúc, đạt JSON valid rate `1.0`.

## 3. Phương pháp thực hiện

### 3.1. Kiến trúc hệ thống và công nghệ sử dụng

Nên chia thành 2 pipeline như báo cáo video QA:

```text
Offline pipeline
PDF giáo trình -> Trích xuất văn bản -> Làm sạch -> Chunking -> Embedding -> ChromaDB + BM25

Online pipeline
Câu hỏi -> Query processing -> Retrieval -> Reranking -> Prompt builder -> LLM -> Answer + Sources
```

File minh chứng:

| Thành phần | File |
|------------|------|
| UI | `app.py`, `ui_components.py`, `styles.css` |
| Cấu hình | `config.py` |
| RAG core | `rag_engine.py` |
| Ingest | `ingest.py`, `ingest_political_6_recursive_bge.py` |
| Tạo đề | `exam_generator.py`, `exam_tab.py` |
| Qwen data | `scripts/generate_qwen_instruction_data.py` |

Công nghệ:

- Python, Streamlit.
- PyMuPDF cho PDF extraction.
- ChromaDB cho vector store.
- BM25/rank-bm25 cho sparse retrieval.
- Sentence-Transformers/Transformers/PyTorch cho embedding và fine-tuning.
- PEFT/TRL/BitsAndBytes cho QLoRA.

### 3.2. Pipeline xử lý và nạp dữ liệu

Mô tả theo các bước:

1. Đầu vào: 6 file PDF giáo trình trong `data/`.
2. Trích xuất văn bản bằng PyMuPDF.
3. Làm sạch văn bản: loại nhiễu PDF, chuẩn hóa khoảng trắng, lọc chunk lỗi.
4. Chunking bằng recursive character splitting với overlap.
5. Sinh embedding bằng model được chọn.
6. Lưu vector vào ChromaDB theo từng môn.
7. Tạo BM25 index song song.
8. Lưu metadata để hiển thị nguồn.

Nên có 1 hình pipeline ingest.

### 3.3. Kỹ thuật lõi: Retrieval-Augmented Generation

Nội dung giống báo cáo TiepLM nhưng thay bằng hệ UniPolis:

- Sparse retrieval bằng BM25.
- Dense retrieval bằng embedding Transformer.
- Hybrid hoặc các mode retrieval trong `rag_engine.py`.
- MMR nếu có dùng để tăng đa dạng ngữ cảnh.
- CrossEncoder/reranker trong mode chất lượng cao.
- Prompt ràng buộc LLM chỉ trả lời dựa trên context.
- Trả về answer và source docs.

Nên viết rõ vì sao RAG phù hợp:

- Giảm hallucination.
- Có nguồn kiểm chứng.
- Có thể mở rộng thêm giáo trình mới.
- Tách retrieval và generation để đánh giá từng phần.

### 3.4. Triển khai chi tiết cho từng chức năng

| Chức năng | Cách triển khai |
|-----------|-----------------|
| Hỏi đáp | Truy hồi top-k chunks, rerank nếu cần, build prompt và gọi LLM. |
| Hiển thị nguồn | Lấy metadata từ document/chunk và hiển thị cùng câu trả lời. |
| Tạo đề/câu hỏi | Dùng `exam_generator.py` để chọn ngữ cảnh và sinh câu hỏi theo loại/độ khó. |
| Pipeline modes | Fast ưu tiên tốc độ; Balance cân bằng; Quality dùng dense/reranker. |
| Fine-tuned retrieval | Dùng model Vietnamese Bi-Encoder V2 như một dense retriever thay thế/so sánh. |

### 3.5. Fine-tuning Vietnamese Bi-Encoder V2

Notebook chính: `Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb`.

Quy trình nên trình bày:

```text
qa_audit.jsonl của LSD/PLDC/TH
  -> gộp dữ liệu
  -> split train/val/test theo (source, chapter)
  -> đánh giá base model
  -> train V1 bằng MultipleNegativesRankingLoss
  -> dùng V1 mine hard negatives
  -> train V2 với mined hard negatives
  -> đánh giá base/V1/V2 trên test độc lập
```

Giải thích quan trọng:

- Dữ liệu có sẵn `train_data.json`, `val_data.json`, `test_data.json` nhưng split này vẫn overlap theo chapter.
- Notebook V2 tự split từ `qa_audit.jsonl` theo `(source, chapter)` để test nghiêm ngặt hơn.
- Test set chỉ dùng cho báo cáo cuối.

### 3.6. QLoRA Qwen2.5-1.5B-Instruct

Notebook chính: `Notebook/fine-tuning-qwen-new.ipynb`.

Script data: `scripts/generate_qwen_instruction_data.py`.

Dataset chính: `data/qwen_lora_instruction_data_clean_no_flashcard/`.

Mô tả task:

| Task | Mục tiêu |
|------|----------|
| `grounded_qa` | Trả lời câu hỏi dựa hoàn toàn vào ngữ cảnh. |
| `question_generation` | Sinh câu hỏi tự luận/ngắn từ đoạn giáo trình. |
| `mcq_generation` | Sinh câu hỏi trắc nghiệm và đáp án. |
| `summary` | Tóm tắt đoạn giáo trình và ý chính. |

Cải tiến dữ liệu:

- Bỏ task `flashcard`.
- Lọc chunk nhiễu, quá ngắn/quá dài, bắt đầu giữa câu.
- Ràng buộc output JSON hợp lệ.
- Dùng test set độc lập.
- Xuất 50 mẫu rubric để chấm thủ công.

## 4. Dữ liệu và độ đo đánh giá

### 4.1. Dữ liệu sử dụng

#### Dữ liệu RAG

| Nhóm dữ liệu | Mô tả |
|--------------|-------|
| 6 PDF giáo trình | Nguồn tri thức chính của hệ thống UniPolis. |
| ChromaDB/BM25 | Dữ liệu đã ingest phục vụ truy hồi. |
| Metadata chunk/page | Dùng để hiển thị nguồn. |

#### Dữ liệu fine-tuning Bi-Encoder

| Nguồn | Vai trò |
|-------|---------|
| `data/training_data/training_data_lsd/qa_audit.jsonl` | QA pairs cho Lịch sử Đảng. |
| `data/training_data/training_data_pldc/qa_audit.jsonl` | QA pairs cho Pháp luật đại cương. |
| `data/training_data/training_data_th/qa_audit.jsonl` | QA pairs cho Triết học Mác-Lênin. |

Split V2:

| Split | Số mẫu |
|-------|------:|
| Train | 2787 |
| Val | 616 |
| Test | 611 |

#### Dữ liệu QLoRA

| Split | Số mẫu |
|-------|------:|
| Train | 1200 |
| Val | 150 |
| Test | 150 |

### 4.2. Độ đo đánh giá retrieval

Nên trình bày như báo cáo video QA:

- `Recall@k`: tỷ lệ truy vấn có đoạn đúng trong top-k.
- `MRR@k`: thứ hạng trung bình nghịch đảo của đoạn đúng đầu tiên.
- `Hit@k` nếu dùng trong một số notebook.
- Latency trung bình nếu so sánh tốc độ.

### 4.3. Độ đo đánh giá generation

Cho Qwen/RAG generation:

- `eval_loss`, `test_loss`: loss trên validation/test.
- `json_valid_rate`: tỷ lệ output parse được JSON.
- `question_rougeL`: độ tương đồng bề mặt cho task sinh câu hỏi.
- Rubric thủ công: đúng schema, bám context, không hallucination, evidence/source hợp lý.

Nếu muốn bổ sung giống báo cáo RAGAS:

- Faithfulness.
- Context Precision.
- Context Recall.
- Answer Correctness.

Nhưng chỉ nên ghi là hướng phát triển nếu bạn chưa chạy RAGAS chính thức.

## 5. Kết quả đánh giá

### 5.1. Kết quả retrieval/embedding

Kết quả artifact cũ:

| Embedding | Recall@1 | Recall@5 | Recall@10 | MRR@5 | Latency trung bình |
|-----------|---------:|---------:|----------:|------:|-------------------:|
| BGE-M3 | 0.7333 | 0.9867 | 0.9867 | 0.8613 | 0.0594s |
| Vietnamese Bi-Encoder FT | 0.6000 | 0.8267 | 0.9867 | 0.7033 | 0.0139s |
| E5-Large | 0.6933 | 0.9467 | 0.9467 | 0.8200 | 0.3748s |

Nhận xét:

- BGE-M3 mạnh nhất về chất lượng trong bảng này.
- Vietnamese Bi-Encoder FT nhanh hơn rõ rệt, có lợi nếu ưu tiên latency.
- Cần phân biệt retrieval benchmark này với notebook Bi-Encoder V2 vì evaluator/split khác nhau.

### 5.2. Kết quả fine-tuning Bi-Encoder

Notebook cũ `bi-encoder-2.ipynb`:

| Model | Split | MRR@10 |
|-------|-------|-------:|
| Base | Val | 0.3809 |
| Fine-tuned | Val | 0.4746 |

Notebook chính `fine-tuning-vietnamese-bi-encoder-v2.ipynb`:

| Model | Val MRR@10 | Test MRR@10 | Delta test so với base |
|-------|-----------:|------------:|-----------------------:|
| Base | 0.4234 | 0.4178 | +0.0000 |
| V1 MNRL | 0.4658 | 0.4795 | +0.0617 |
| V2 Hard Negative | 0.4864 | 0.4982 | +0.0805 |

Kết luận:

- Fine-tuning cải thiện retrieval.
- Hard negative mining cải thiện thêm so với V1.
- V2 đáng tin hơn bản cũ vì có test set độc lập theo chapter-disjoint split.

### 5.3. Kết quả QLoRA Qwen

| Chỉ số | Giá trị |
|--------|--------:|
| Training loss cuối | 0.0507 |
| Validation loss | 0.1441 |
| Test loss | 0.0807 |
| JSON valid rate | 1.0000 |
| Question ROUGE-L | 0.4922 |
| Số mẫu generation eval | 80 |

So với lần chạy trước:

| Chỉ số | Lần trước | Lần mới |
|--------|----------:|--------:|
| Validation loss | 0.2861 | 0.1441 |
| JSON valid rate | 0.6000 | 1.0000 |
| Question ROUGE-L | 0.6074 | 0.4922 |

Nhận xét:

- JSON validity cải thiện mạnh từ `0.6` lên `1.0`.
- Validation loss giảm đáng kể.
- ROUGE-L giảm nhưng không nên xem là thất bại vì câu hỏi đúng có thể khác wording.
- Cần chấm thủ công 50 mẫu rubric để hoàn thiện đánh giá.

### 5.4. Kết quả demo định tính

Nên đưa 3-5 ví dụ:

| Loại ví dụ | Nội dung nên chụp/ghi |
|------------|-----------------------|
| Câu hỏi khái niệm | Hỏi định nghĩa/phạm trù và câu trả lời có nguồn. |
| Câu hỏi so sánh | Hệ thống tổng hợp nhiều ý từ giáo trình. |
| Câu hỏi ngoài phạm vi | Hệ thống từ chối hoặc báo thiếu thông tin. |
| Tạo đề/câu hỏi | Ảnh tab exam/câu hỏi được sinh. |
| Lỗi còn tồn tại | Ví dụ chunk nhiễu hoặc câu trả lời chưa gọn. |

## 6. Những cải tiến và đóng góp học sâu

Phần này học theo báo cáo TiepLM có mục “Những cải tiến mới so với bài trình bày trước”. Với UniPolis, nên trình bày thành 3 cải tiến.

### 6.1. Cải tiến retrieval bằng fine-tuning Bi-Encoder

Nội dung:

- Bản đầu `bi-encoder-2.ipynb` chỉ có train/val.
- Bản V2 bổ sung test set độc lập theo `(source, chapter)`.
- Bản V2 thêm hard negative mining.
- Kết quả V2 test MRR@10 đạt `0.4982`.

### 6.2. Cải tiến generation bằng QLoRA Qwen

Nội dung:

- Sinh instruction data từ 6 giáo trình.
- Bỏ flashcard để tập trung task chính.
- Làm sạch dữ liệu và ràng buộc JSON schema.
- Fine-tune Qwen2.5-1.5B-Instruct bằng QLoRA trên Kaggle T4.
- JSON valid rate đạt `1.0`.

### 6.3. Hoàn thiện khả năng tái lập

Nội dung:

- Notebook riêng cho Bi-Encoder V2.
- Notebook riêng cho Qwen QLoRA.
- Script sinh data Qwen.
- Artifact đánh giá và rubric.
- App Streamlit chạy bằng `streamlit run app.py`.

## 7. Kết luận và hướng phát triển

### 7.1. Kết quả đạt được

- Xây dựng hệ thống RAG hoạt động trên 6 giáo trình.
- Có giao diện hỏi đáp/tạo đề.
- Có so sánh retrieval và metric định lượng.
- Có fine-tuning Bi-Encoder cải thiện retrieval.
- Có QLoRA LLM nhỏ cho sinh JSON học tập có cấu trúc.

### 7.2. Hạn chế

- PDF còn nhiễu, một số chunk bị đứt câu.
- Fine-tuning Bi-Encoder mới dùng 3 môn có training data, chưa đủ 6 môn.
- Qwen output vẫn phụ thuộc chất lượng chunk.
- RAG end-to-end chưa có RAGAS hoặc human evaluation đủ lớn.
- Qwen adapter chưa chắc đã tích hợp trực tiếp vào app nếu chưa triển khai inference.

### 7.3. Hướng phát triển

- Mở rộng training data Bi-Encoder cho đủ 6 môn.
- Chấm thủ công 50 mẫu Qwen theo rubric.
- Cải thiện metadata chương/mục/trang.
- Thêm RAGAS hoặc LLM-as-Judge cho end-to-end QA.
- Tích hợp Qwen LoRA vào module tạo câu hỏi nếu có hạ tầng inference.

## 8. Bảng phân công

Nếu làm nhóm, dùng bảng giống báo cáo mẫu:

| Nhiệm vụ | Thành viên 1 | Thành viên 2 | Thành viên 3 | Thành viên 4 |
|----------|--------------|--------------|--------------|--------------|
| Tìm hiểu RAG và tài liệu liên quan | X | X | X | X |
| Xây dựng ingest/RAG engine | X | | | |
| Giao diện Streamlit | | X | | |
| Fine-tuning Bi-Encoder | | | X | |
| QLoRA Qwen và instruction data | | | | X |
| Thực nghiệm và đánh giá | X | X | X | X |
| Viết báo cáo/slide | X | X | X | X |

Nếu làm cá nhân, đổi thành bảng nhiệm vụ/thời gian/trạng thái.

## Tài liệu tham khảo gợi ý

Nên có các nhóm tài liệu:

- RAG: Retrieval-Augmented Generation.
- BM25 / Information Retrieval.
- Sentence-BERT / Sentence-Transformers.
- LoRA, QLoRA, PEFT.
- Qwen2.5 technical/model card nếu cần.
- ChromaDB/LangChain/Sentence-Transformers docs nếu báo cáo cho phép tài liệu kỹ thuật.

## Phụ lục

### Phụ lục A - File minh chứng

| File/thư mục | Vai trò |
|--------------|---------|
| `app.py` | Streamlit app |
| `rag_engine.py` | RAG engine |
| `config.py` | Cấu hình môn học/pipeline |
| `ingest.py` | Ingest lõi |
| `ingest_political_6_recursive_bge.py` | Ingest 6 giáo trình |
| `exam_generator.py`, `exam_tab.py` | Tạo đề/câu hỏi |
| `Notebook/compare_new.ipynb` | So sánh retrieval |
| `Notebook/bi-encoder-2.ipynb` | Fine-tune Bi-Encoder bản đầu |
| `Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb` | Bi-Encoder V2 + hard negative mining |
| `scripts/generate_qwen_instruction_data.py` | Sinh data Qwen |
| `Notebook/fine-tuning-qwen-new.ipynb` | QLoRA Qwen kết quả mới |
| `data/qwen_lora_instruction_data_clean_no_flashcard/` | Dataset Qwen chính |

### Phụ lục B - Checklist hình/bảng nên có

- Hình kiến trúc UniPolis.
- Hình offline ingest pipeline.
- Hình online RAG pipeline.
- Bảng 6 giáo trình.
- Bảng retrieval baseline.
- Bảng Bi-Encoder V2.
- Bảng QLoRA Qwen.
- Ảnh demo Streamlit.
- Ảnh ví dụ source/citation.
- Ảnh hoặc bảng rubric Qwen 50 mẫu nếu đã chấm.
