# Đặc tả đồ án CS431: UniPolis - Hệ thống RAG hỗ trợ học tập các môn lý luận chính trị

## 1. Thông tin chung

- **Tên đề tài:** UniPolis - Hệ thống hỏi đáp và tạo đề thi dựa trên RAG cho giáo trình lý luận chính trị.
- **Môn học:** CS431 - Các Kỹ thuật học sâu và ứng dụng.
- **Loại bài toán:** Truy hồi thông tin ngữ nghĩa, sinh câu trả lời có căn cứ, sinh câu hỏi/đề thi từ tài liệu.
- **Ngôn ngữ xử lý:** Tiếng Việt.
- **Sản phẩm cuối:** Ứng dụng Streamlit cho phép chọn môn học, đặt câu hỏi, xem câu trả lời kèm nguồn trích dẫn và tạo đề thi dựa trên nội dung giáo trình.

## 2. Bối cảnh và vấn đề

Các giáo trình lý luận chính trị ở bậc đại học thường có dung lượng lớn, nhiều khái niệm trừu tượng, nhiều mốc sự kiện và yêu cầu trích dẫn chính xác. Sinh viên khi ôn tập thường gặp khó khăn trong việc tìm đúng đoạn nội dung, tổng hợp kiến thức và tự kiểm tra mức độ hiểu bài.

Các mô hình ngôn ngữ lớn có khả năng trả lời tự nhiên nhưng nếu sử dụng trực tiếp có thể sinh thông tin không có trong giáo trình. Vì vậy, đồ án xây dựng một hệ thống **Retrieval-Augmented Generation (RAG)** để neo câu trả lời vào nguồn tài liệu cụ thể, giảm hiện tượng hallucination và hỗ trợ học tập theo từng môn.

## 3. Mục tiêu đồ án

### 3.1. Mục tiêu tổng quát

Xây dựng hệ thống trợ lý học tập tiếng Việt có khả năng truy xuất nội dung từ giáo trình lý luận chính trị, sinh câu trả lời dựa trên ngữ cảnh đã truy hồi và hiển thị nguồn tham chiếu để người học kiểm chứng.

### 3.2. Mục tiêu cụ thể

- Xử lý PDF giáo trình thành các đoạn văn bản có thể truy hồi.
- Xây dựng chỉ mục truy hồi gồm vector store ChromaDB và chỉ mục BM25.
- Áp dụng các mô hình embedding như `BAAI/bge-m3`, `intfloat/multilingual-e5-large` và Vietnamese Bi-Encoder fine-tuned.
- Thiết kế các chế độ truy hồi: nhanh, cân bằng và chất lượng cao.
- Tích hợp CrossEncoder reranker để sắp xếp lại các đoạn liên quan trong chế độ chất lượng.
- Sinh câu trả lời tiếng Việt bằng LLM dựa trên ngữ cảnh được cung cấp.
- Hiển thị nguồn trích dẫn theo trang/chunk để tăng tính kiểm chứng.
- Hỗ trợ tạo đề thi gồm nhiều dạng câu hỏi dựa trên nội dung truy hồi.
- Đánh giá retrieval bằng các chỉ số Recall@k và MRR@k.
- Fine-tune LLM nhỏ bằng QLoRA để sinh câu hỏi, tóm tắt và trả lời có cấu trúc JSON dựa trên ngữ cảnh giáo trình.

## 4. Phạm vi dữ liệu

Hệ thống hướng đến 6 giáo trình lý luận chính trị:

| STT | Môn học | Mục đích sử dụng |
|-----|---------|------------------|
| 1 | Lịch sử Đảng Cộng sản Việt Nam | Hỏi đáp, ôn tập sự kiện và đường lối |
| 2 | Pháp luật đại cương | Hỏi đáp khái niệm, quy định, phân biệt thuật ngữ |
| 3 | Triết học Mác-Lênin | Hỏi đáp khái niệm, quy luật, phạm trù triết học |
| 4 | Kinh tế Chính trị Mác-Lênin | Hỏi đáp lý thuyết kinh tế chính trị |
| 5 | Chủ nghĩa Xã hội Khoa học | Hỏi đáp lý luận về CNXH và giai cấp công nhân |
| 6 | Tư tưởng Hồ Chí Minh | Hỏi đáp quan điểm, tư tưởng và phương pháp luận |

Dữ liệu đầu vào là các file PDF trong thư mục `data/`. Sau khi xử lý, mỗi môn có một thư mục ChromaDB và BM25 riêng để giảm nhiễu giữa các miền kiến thức.

## 5. Bài toán học sâu trong đồ án

Đồ án sử dụng và đánh giá các kỹ thuật học sâu trong các thành phần sau:

- **Sentence Embedding:** chuyển câu hỏi và đoạn văn bản thành vector ngữ nghĩa bằng các mô hình Transformer.
- **Bi-Encoder Retrieval:** truy hồi nhanh các đoạn có ngữ nghĩa gần câu hỏi.
- **Fine-tuning Bi-Encoder:** tinh chỉnh Vietnamese Bi-Encoder trên các cặp query-document thuộc miền giáo trình chính trị.
- **CrossEncoder Reranking:** đánh giá trực tiếp cặp câu hỏi-đoạn văn để sắp xếp lại kết quả truy hồi.
- **LLM Generation:** sinh câu trả lời và đề thi dựa trên các đoạn ngữ cảnh đã truy hồi.
- **QLoRA / Supervised Fine-tuning:** tinh chỉnh `Qwen2.5-1.5B-Instruct` trên instruction data sinh từ giáo trình để tạo câu hỏi, tóm tắt, câu hỏi trắc nghiệm và trả lời có căn cứ.

## 6. Kiến trúc hệ thống

Luồng tổng thể của hệ thống:

```text
PDF giáo trình
  -> Trích xuất văn bản bằng PyMuPDF
  -> Làm sạch văn bản
  -> Recursive Character Chunking
  -> Tạo embedding bằng mô hình Transformer
  -> Lưu vector vào ChromaDB
  -> Tạo BM25 index
  -> Người dùng đặt câu hỏi trên Streamlit
  -> Retrieval bằng BM25 / Dense MMR / Hybrid
  -> Tùy chọn rerank bằng CrossEncoder
  -> LLM sinh câu trả lời dựa trên ngữ cảnh
  -> Hiển thị câu trả lời và nguồn trích dẫn
```

### 6.1. Thành phần chính

| Thành phần | File liên quan | Vai trò |
|------------|----------------|--------|
| Giao diện người dùng | `app.py`, `ui_components.py`, `styles.css` | Chat, chọn môn, chọn pipeline, hiển thị nguồn |
| Cấu hình hệ thống | `config.py` | Danh sách môn học, pipeline mode, câu hỏi mẫu |
| RAG engine | `rag_engine.py` | Truy hồi, rerank, tạo prompt và gọi LLM |
| Ingest dữ liệu | `ingest.py`, `ingest_political_6_recursive_bge.py` | Đọc PDF, chunking, embedding, tạo ChromaDB và BM25 |
| Tạo đề thi | `exam_generator.py`, `exam_tab.py` | Sinh câu hỏi theo chủ đề, độ khó và loại câu |
| Fine-tuning | `Notebook/fine-tuning-vietnamese-bi-encoder.ipynb` | Huấn luyện mô hình bi-encoder tiếng Việt |
| Đánh giá | `Notebook/compare_new.ipynb`, `artifacts/*.csv` | So sánh embedding/retrieval bằng metric |

## 7. Thiết kế pipeline truy hồi

Hệ thống có nhiều chế độ pipeline để cân bằng giữa tốc độ và chất lượng:

| Chế độ | Cấu hình | Mục tiêu |
|--------|----------|----------|
| Fast | BM25, top-k thấp, không rerank | Phản hồi nhanh |
| Balance | BM25, top-k lớn hơn, không rerank | Cân bằng tốc độ và độ phủ |
| Quality | Dense MMR + BGE reranker | Tăng chất lượng và độ đa dạng ngữ cảnh |
| Quality - VN Bi-Encoder FT | Dense MMR bằng model fine-tuned + reranker | Thử nghiệm embedding đã tinh chỉnh cho miền dữ liệu |

Chiến lược chunking mặc định trong script ingest batch dùng `RecursiveCharacterTextSplitter` với `chunk_size=512`, `chunk_overlap=64` và các separator `\n\n`, `\n`, `.`, khoảng trắng.

## 8. Mô hình và công nghệ sử dụng

| Nhóm | Công nghệ |
|------|-----------|
| Ngôn ngữ lập trình | Python |
| Deep learning | PyTorch, Transformers, Sentence-Transformers |
| Embedding | BGE-M3, Multilingual E5-Large, Vietnamese Bi-Encoder |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| LLM fine-tuning | Qwen2.5-1.5B-Instruct, PEFT LoRA/QLoRA, TRL SFTTrainer |
| Vector database | ChromaDB |
| Sparse retrieval | BM25, `rank-bm25` |
| Orchestration | LangChain |
| LLM | Gemini / OpenAI tùy module cấu hình, Qwen2.5-1.5B-Instruct fine-tuned cho thí nghiệm sinh có cấu trúc |
| Giao diện | Streamlit |
| Xử lý PDF | PyMuPDF |
| NLP tiếng Việt | underthesea |

## 9. Chức năng hệ thống

### 9.1. Chức năng bắt buộc

- Chọn giáo trình/môn học cần hỏi đáp.
- Đặt câu hỏi tự nhiên bằng tiếng Việt.
- Trả lời dựa trên nội dung giáo trình đã ingest.
- Hiển thị các đoạn nguồn được dùng để trả lời.
- Chọn chế độ pipeline truy hồi.
- Điều chỉnh số lượng đoạn truy hồi top-k.
- Xóa lịch sử chat và tái tạo câu trả lời.

### 9.2. Chức năng mở rộng

- Tạo đề thi theo môn học hoặc chủ đề.
- Chọn số lượng câu hỏi, độ khó và phân bổ loại câu.
- Xuất lịch sử hội thoại sang Markdown hoặc TXT.
- So sánh hiệu quả giữa các mô hình embedding.
- Fine-tune Vietnamese Bi-Encoder cho miền giáo trình chính trị.

## 10. Yêu cầu phi chức năng

- **Tính đúng nguồn:** câu trả lời chỉ dựa trên ngữ cảnh được truy hồi.
- **Tính kiểm chứng:** mỗi câu trả lời cần hiển thị nguồn hoặc trang liên quan nếu metadata có sẵn.
- **Hiệu năng:** chế độ Fast/Balance phải đủ nhanh để dùng tương tác trên máy cá nhân.
- **Khả năng mở rộng:** có thể thêm môn học mới bằng cách ingest PDF và bổ sung cấu hình trong `SUBJECT_CATALOGUE`.
- **Bảo mật:** API key phải đặt trong `.env`, không hard-code trong mã nguồn.
- **Tính tái lập:** quá trình ingest, fine-tuning và đánh giá cần có notebook/script tương ứng.

## 11. Phương pháp fine-tuning Bi-Encoder

Mô hình cơ sở được định hướng là `bkai-foundation-models/vietnamese-bi-encoder`. Dữ liệu huấn luyện gồm các cặp câu hỏi và đoạn văn bản liên quan được sinh từ giáo trình, có thể kết hợp dữ liệu tự động và dữ liệu kiểm tra thủ công.

Quy trình dự kiến:

```text
PDF giáo trình
  -> Chunk theo đoạn/chủ đề
  -> Sinh query-document pairs
  -> Lọc dữ liệu nhiễu
  -> Chia train/validation/test
  -> Fine-tune bằng contrastive learning
  -> Lưu model tại models/vietnamese-bi-encoder-finetuned
  -> Rebuild vector store bằng model đã fine-tune
  -> Đánh giá Recall@k và MRR@k
```

Loss function phù hợp là `MultipleNegativesRankingLoss` vì bài toán cần kéo gần vector của câu hỏi và đoạn đúng, đồng thời đẩy xa các đoạn không liên quan trong batch.

## 12. Đánh giá thực nghiệm

### 12.1. Đánh giá retrieval

Các chỉ số chính:

- **Recall@k:** tỷ lệ câu hỏi có đoạn đúng xuất hiện trong top-k kết quả.
- **MRR@k:** đo thứ hạng trung bình của đoạn đúng trong top-k.
- **Latency:** thời gian truy hồi trung bình.

Ví dụ kết quả hiện có trong `artifacts/compare_retrieval_20260514_224616.csv`:

| Embedding | Recall@1 | Recall@5 | Recall@10 | MRR@5 | Latency trung bình |
|-----------|----------|----------|-----------|-------|--------------------|
| BGE-M3 | 0.7333 | 0.9867 | 0.9867 | 0.8613 | 0.0594s |
| Vietnamese Bi-Encoder FT | 0.6000 | 0.8267 | 0.9867 | 0.7033 | 0.0139s |
| E5-Large | 0.6933 | 0.9467 | 0.9467 | 0.8200 | 0.3748s |

### 12.2. Đánh giá QLoRA Qwen2.5-1.5B-Instruct

Thí nghiệm QLoRA sử dụng instruction data sinh từ 6 giáo trình, đã lọc nhiễu PDF và loại bỏ task flashcard để tập trung vào các tác vụ có ích cho hệ thống học tập: `grounded_qa`, `question_generation`, `mcq_generation` và `summary`. Bộ dữ liệu được chia thành train/validation/test độc lập với kích thước `1200/150/150` mẫu. Notebook chính: `Notebook/fine-tuning-qwen-new.ipynb`.

Kết quả chạy trên Kaggle T4:

| Chỉ số | Giá trị |
|--------|--------:|
| Training loss cuối | 0.0507 |
| Validation loss | 0.1441 |
| Test loss | 0.0807 |
| JSON valid rate | 1.0000 |
| Question ROUGE-L | 0.4922 |
| Số mẫu generation eval | 80 |

So với lần chạy trước, validation loss giảm từ `0.2861` xuống `0.1441` và tỷ lệ JSON hợp lệ tăng từ `0.6` lên `1.0`. Điều này cho thấy việc làm sạch dữ liệu, bỏ flashcard và ràng buộc output JSON trong prompt giúp mô hình học tốt hơn định dạng sinh có cấu trúc. ROUGE-L của task sinh câu hỏi giảm so với lần trước, nhưng đây chỉ là metric bề mặt theo chuỗi; câu hỏi có thể khác cách diễn đạt nhưng vẫn đúng nội dung nên cần chấm thủ công bằng rubric.

Notebook cũng xuất 50 mẫu test để đánh giá thủ công:

- `manual_rubric_50_test_samples.jsonl`
- `manual_rubric_50_test_samples.csv`

### 12.3. Đánh giá định tính

Mỗi môn học cần chọn một tập câu hỏi mẫu gồm:

- Câu hỏi khái niệm.
- Câu hỏi so sánh/phân biệt.
- Câu hỏi yêu cầu liệt kê ý chính.
- Câu hỏi theo mốc lịch sử hoặc văn bản pháp luật.
- Câu hỏi ngoài phạm vi giáo trình để kiểm tra khả năng từ chối trả lời.

Tiêu chí đánh giá định tính:

- Câu trả lời có bám sát giáo trình không.
- Có nêu rõ khi thiếu thông tin không.
- Nguồn trích dẫn có liên quan không.
- Câu trả lời có cấu trúc, dễ hiểu, phù hợp tiếng Việt học thuật không.

## 13. Tiêu chí nghiệm thu

Đồ án được xem là hoàn thành khi đáp ứng các tiêu chí sau:

- Chạy được ứng dụng bằng `streamlit run app.py`.
- Ít nhất 3 môn học có dữ liệu ChromaDB và BM25 có thể truy vấn ổn định.
- Hệ thống trả lời được câu hỏi tiếng Việt và hiển thị nguồn tham khảo.
- Có ít nhất 2 chế độ pipeline hoạt động để so sánh tốc độ/chất lượng.
- Có bảng đánh giá retrieval bằng Recall@k và MRR@k.
- Có mô tả rõ quy trình fine-tuning hoặc kết quả fine-tuning Vietnamese Bi-Encoder.
- Có mô tả rõ quy trình QLoRA/SFT LLM nhỏ và kết quả đánh giá trên validation/test set.
- Có báo cáo phân tích hạn chế và hướng phát triển.

## 14. Kế hoạch thực hiện

| Giai đoạn | Công việc | Kết quả |
|-----------|-----------|---------|
| 1 | Khảo sát dữ liệu và yêu cầu | Danh sách giáo trình, phạm vi câu hỏi |
| 2 | Xây dựng pipeline ingest | ChromaDB và BM25 cho từng môn |
| 3 | Xây dựng RAG engine | API hỏi đáp trả về answer và source docs |
| 4 | Phát triển giao diện Streamlit | Demo tương tác với người dùng |
| 5 | Fine-tune và/hoặc so sánh embedding | Model FT và bảng metric |
| 6 | Tích hợp tạo đề thi | Tab sinh đề theo chủ đề/độ khó |
| 7 | Đánh giá và viết báo cáo | Bảng kết quả, phân tích, kết luận |

## 15. Rủi ro và hướng xử lý

| Rủi ro | Ảnh hưởng | Hướng xử lý |
|--------|-----------|-------------|
| PDF có nhiễu, lỗi xuống dòng, lỗi mục lục | Chunk kém chất lượng | Làm sạch bằng regex, lọc trang/chunk quá ngắn |
| Retrieval lấy sai đoạn | Câu trả lời sai hoặc thiếu | Tăng top-k, dùng Dense MMR, reranker, cải thiện chunking |
| Fine-tuned model không vượt baseline | Kết quả thực nghiệm không như kỳ vọng | Phân tích dữ liệu train, hard negatives, so sánh công bằng theo latency |
| QLoRA học tốt định dạng nhưng nội dung còn phụ thuộc chunk nhiễu | Output JSON hợp lệ nhưng có thể chứa câu bị đứt hoặc đoạn dài | Tiếp tục lọc dữ liệu, chấm thủ công rubric, bổ sung post-processing hoặc constrained decoding |
| LLM hallucination | Mất tính tin cậy | Prompt bắt buộc chỉ trả lời theo ngữ cảnh, hiển thị nguồn |
| Chi phí/tốc độ inference cao | Trải nghiệm kém | Cung cấp chế độ Fast/Balance dùng BM25 không rerank |

## 16. Hướng phát triển

- Bổ sung đánh giá end-to-end bằng RAGAS hoặc đánh giá thủ công có rubric.
- Cải thiện metadata theo chương, mục, tiểu mục thay vì chỉ theo trang/chunk.
- Thêm hard negative mining cho quá trình fine-tuning bi-encoder.
- Xây dựng bộ câu hỏi kiểm thử chuẩn cho đủ 6 môn.
- Hỗ trợ tìm kiếm liên môn có kiểm soát bằng metadata filtering.
- Thêm chế độ học tập: flashcard, tóm tắt chương, lộ trình ôn tập.
- Đóng gói ứng dụng bằng Docker để dễ triển khai.

## 17. Kết luận

UniPolis là một đồ án phù hợp với môn CS431 vì kết hợp nhiều kỹ thuật học sâu hiện đại trong NLP: embedding Transformer, bi-encoder retrieval, fine-tuning bằng contrastive learning, CrossEncoder reranking và LLM generation. Hệ thống không chỉ là ứng dụng hỏi đáp, mà còn là một pipeline RAG hoàn chỉnh có dữ liệu, mô hình, đánh giá và giao diện demo, đáp ứng yêu cầu của một đồ án ứng dụng học sâu trong thực tế.
