# UniPolis - Hệ thống RAG hỗ trợ hỏi đáp giáo trình lý luận chính trị tiếng Việt

**Môn học:** CS431 - Các kỹ thuật học sâu và ứng dụng  
**Giảng viên hướng dẫn:** TS. Nguyễn Vinh Tiệp  
**Sinh viên thực hiện:** [Điền thông tin sinh viên]  
**Thời gian:** [Điền học kỳ/năm học]

## Mục lục

1. Giới thiệu đề tài
2. Phương pháp thực hiện
3. Dữ liệu và độ đo đánh giá
4. Kết quả đánh giá
5. Kết luận và hướng phát triển
6. Bảng phân công
7. Tài liệu tham khảo
8. Phụ lục

## 1. Giới thiệu đề tài

### 2.1. Bối cảnh và vấn đề

Các môn lý luận chính trị là nhóm môn học bắt buộc trong chương trình đại học. Nội dung các giáo trình thường dài, có văn phong học thuật, chứa nhiều khái niệm, phạm trù, luận điểm và mốc lịch sử. Khi ôn tập, sinh viên thường cần trả lời các câu hỏi như: một khái niệm được định nghĩa như thế nào, một sự kiện có ý nghĩa gì, hai quan điểm khác nhau ra sao, hoặc một luận điểm thuộc chương nào trong giáo trình.

Các phương pháp tra cứu truyền thống như tìm kiếm từ khóa trong PDF có nhiều hạn chế. Người học phải biết chính xác từ khóa cần tìm, trong khi câu hỏi tự nhiên thường được diễn đạt khác với văn bản giáo trình. Ngược lại, các mô hình ngôn ngữ lớn có thể trả lời mạch lạc nhưng không đảm bảo câu trả lời dựa đúng vào giáo trình. Điều này đặc biệt rủi ro với các môn học yêu cầu tính chính xác và khả năng kiểm chứng nguồn.

Vì vậy, bài toán đặt ra là xây dựng một hệ thống hỏi đáp có khả năng hiểu câu hỏi tiếng Việt, truy xuất đúng đoạn giáo trình liên quan và sinh câu trả lời dựa trên nguồn được cung cấp. Hệ thống cũng cần hiển thị nguồn tham chiếu để người học có thể kiểm chứng lại nội dung.

### 2.2. Mục tiêu và giải pháp

Mục tiêu tổng quát của đồ án là xây dựng UniPolis, một trợ lý học tập tiếng Việt dựa trên RAG cho 6 giáo trình lý luận chính trị. Hệ thống hướng đến việc hỗ trợ sinh viên tra cứu, hỏi đáp và ôn tập nội dung giáo trình một cách nhanh chóng, có căn cứ và dễ kiểm chứng.

Các mục tiêu cụ thể gồm:

- Xử lý PDF giáo trình thành các đoạn văn bản có thể truy hồi.
- Xây dựng chỉ mục sparse retrieval bằng BM25 và dense retrieval bằng embedding Transformer.
- Thiết kế các chế độ pipeline retrieval để cân bằng tốc độ và chất lượng.
- Sinh câu trả lời dựa trên ngữ cảnh truy hồi, hạn chế hallucination.
- Hiển thị nguồn tham chiếu theo chunk/trang nếu metadata có sẵn.
- Fine-tune Vietnamese Bi-Encoder cho miền giáo trình lý luận chính trị.
- So sánh nhiều mô hình embedding bằng các độ đo retrieval.
- Thử nghiệm QLoRA LLM nhỏ cho sinh nội dung học tập có cấu trúc.

## 3. Phương pháp thực hiện

### 3.1. Kiến trúc hệ thống

UniPolis được thiết kế theo hai giai đoạn: offline pipeline và online pipeline.

Offline pipeline chịu trách nhiệm xử lý dữ liệu giáo trình trước khi người dùng truy vấn:

```text
PDF giáo trình
  -> Trích xuất văn bản bằng PyMuPDF
  -> Làm sạch văn bản
  -> Recursive character chunking
  -> Tạo embedding
  -> Lưu ChromaDB
  -> Tạo BM25 index
```

Online pipeline được kích hoạt khi người dùng đặt câu hỏi:

```text
Câu hỏi người dùng
  -> Chọn môn học và pipeline mode
  -> Truy hồi BM25 / Dense / Dense MMR / Hybrid
  -> Rerank nếu bật chế độ Quality
  -> Build prompt với context
  -> LLM sinh câu trả lời
  -> Hiển thị answer và source docs
```

Các thành phần chính của hệ thống:


| Thành phần             | File liên quan                                        | Vai trò                                          |
| ---------------------- | ----------------------------------------------------- | ------------------------------------------------ |
| Giao diện              | `app.py`, `ui_components.py`, `styles.css`            | Chat UI, chọn môn, chọn pipeline, hiển thị nguồn |
| Cấu hình               | `config.py`                                           | Danh sách môn học, pipeline mode, preset câu hỏi |
| RAG engine             | `rag_engine.py`                                       | Truy hồi, rerank, build prompt, gọi LLM          |
| Ingest dữ liệu         | `ingest.py`, `ingest_political_6_recursive_bge.py`    | Đọc PDF, chunking, embedding, ChromaDB, BM25     |
| Tạo đề/câu hỏi         | `exam_generator.py`, `exam_tab.py`                    | Sinh câu hỏi ôn tập từ nội dung giáo trình       |
| Fine-tuning Bi-Encoder | `Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb` | Train và đánh giá Vietnamese Bi-Encoder V2       |
| QLoRA Qwen             | `Notebook/fine-tuning-qwen-new.ipynb`                 | Fine-tune LLM nhỏ cho generation có cấu trúc     |


### 3.2. Dữ liệu giáo trình và ingest

Hệ thống sử dụng 6 giáo trình lý luận chính trị trong thư mục `data/`:


| STT | Môn học                        | File PDF                                        |
| --- | ------------------------------ | ----------------------------------------------- |
| 1   | Lịch sử Đảng Cộng sản Việt Nam | `lich-su-dang-cong-san-viet-nam-giao-trinh.pdf` |
| 2   | Pháp luật đại cương            | `phap-luat-dai-cuong-giao-trinh.pdf`            |
| 3   | Triết học Mác-Lênin            | `triet-hoc-mac-lenin-giao-trinh.pdf`            |
| 4   | Kinh tế Chính trị Mác-Lênin    | `kinh-te-chinh-tri-mac-lenin-giao-trinh.pdf`    |
| 5   | Chủ nghĩa Xã hội Khoa học      | `chu-nghia-xa-hoi-khoa-hoc-giao-trinh.pdf`      |
| 6   | Tư tưởng Hồ Chí Minh           | `tu-tuong-hcm-giao-trinh.pdf`                   |


Sau khi trích xuất văn bản từ PDF, hệ thống làm sạch nội dung, loại bỏ các trang hoặc đoạn quá ngắn, sau đó chia văn bản bằng `RecursiveCharacterTextSplitter` với `chunk_size = 512` và `chunk_overlap = 64`. Mỗi chunk được lưu cùng metadata để phục vụ truy hồi và hiển thị nguồn.

### 3.3. Retrieval-Augmented Generation

RAG là kỹ thuật cốt lõi của UniPolis. Thay vì yêu cầu LLM trả lời trực tiếp từ kiến thức tham số, hệ thống trước tiên truy xuất các đoạn giáo trình liên quan rồi đưa các đoạn đó vào prompt. Cách làm này giúp giảm hallucination và tăng khả năng kiểm chứng.

Các phương pháp truy hồi trong hệ thống gồm:

- BM25: truy hồi dựa trên từ khóa, phù hợp với tên riêng, thuật ngữ hoặc câu hỏi có từ khóa rõ ràng.
- Dense retrieval: truy hồi dựa trên embedding, phù hợp với câu hỏi diễn đạt tự nhiên.
- Dense MMR: truy hồi dense có xét đa dạng kết quả, tránh lấy nhiều đoạn quá giống nhau.
- Reranking: dùng CrossEncoder để chấm lại cặp câu hỏi và đoạn văn, giúp sắp xếp top-k chính xác hơn.

Các pipeline mode trong hệ thống:


| Mode                       | Cấu hình                                                   | Mục tiêu                          |
| -------------------------- | ---------------------------------------------------------- | --------------------------------- |
| Fast                       | BM25, top-k thấp, không rerank                             | Phản hồi nhanh                    |
| Balance                    | BM25, top-k cao hơn, không rerank                          | Cân bằng tốc độ và độ phủ         |
| Quality                    | Dense MMR + BGE reranker                                   | Tăng chất lượng truy hồi          |
| Quality - VN Bi-Encoder FT | Dense MMR bằng Vietnamese Bi-Encoder fine-tuned + reranker | Thử nghiệm retriever đã fine-tune |


Prompt của hệ thống yêu cầu LLM chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu ngữ cảnh không đủ thông tin, hệ thống phải nói rõ không tìm thấy thông tin trong giáo trình.

### 3.4. Fine-tuning Vietnamese Bi-Encoder

Để tăng tính học sâu cho thành phần retrieval, đồ án fine-tune mô hình `bkai-foundation-models/vietnamese-bi-encoder`. Đây là một mô hình embedding tiếng Việt, phù hợp cho bài toán chuyển câu hỏi và đoạn văn bản thành vector ngữ nghĩa.

Notebook chính là `Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb`. Dữ liệu huấn luyện được lấy từ các file `qa_audit.jsonl` của 3 môn:

- Lịch sử Đảng Cộng sản Việt Nam.
- Pháp luật đại cương.
- Triết học Mác-Lênin.

Quy trình fine-tuning:

```text
qa_audit.jsonl
  -> gộp dữ liệu 3 môn
  -> split train/val/test theo (source, chapter)
  -> đánh giá base model
  -> train V1 bằng MultipleNegativesRankingLoss
  -> dùng V1 mine hard negatives
  -> train V2 với mined hard negatives
  -> đánh giá base/V1/V2 trên test set độc lập
```

Việc split theo `(source, chapter)` giúp hạn chế leakage giữa train, validation và test. Mặc dù dataset có sẵn `train_data.json`, `val_data.json`, `test_data.json`, notebook V2 không dùng trực tiếp các split này vì chúng vẫn có overlap theo chapter. Thay vào đó, notebook tự split lại từ `qa_audit.jsonl` để test set độc lập hơn.

Loss được sử dụng là `MultipleNegativesRankingLoss`. Với mỗi query, positive passage đúng được kéo gần trong không gian embedding, trong khi các positive của query khác trong batch đóng vai trò negative. Ở vòng V2, hard negative mining được sử dụng để tìm các đoạn dễ nhầm, giúp mô hình học khả năng phân biệt tốt hơn.

### 3.5. QLoRA Qwen2.5-1.5B-Instruct

Ngoài RAG Q&A, đồ án thử nghiệm thêm QLoRA cho mô hình `Qwen2.5-1.5B-Instruct` nhằm sinh nội dung học tập có cấu trúc. Phần này được xem là mở rộng, không thay thế trọng tâm chính là RAG retrieval.

Script sinh dữ liệu: `scripts/generate_qwen_instruction_data.py`.  
Notebook huấn luyện: `Notebook/fine-tuning-qwen-new.ipynb`.  
Dataset chính: `data/qwen_lora_instruction_data_clean_no_flashcard/`.

Các task gồm:


| Task                  | Mục tiêu                             |
| --------------------- | ------------------------------------ |
| `grounded_qa`         | Trả lời dựa trên ngữ cảnh giáo trình |
| `question_generation` | Sinh câu hỏi từ đoạn giáo trình      |
| `mcq_generation`      | Sinh câu hỏi trắc nghiệm             |
| `summary`             | Tóm tắt nội dung giáo trình          |


Các cải tiến dữ liệu gồm bỏ task `flashcard`, lọc chunk nhiễu, yêu cầu JSON hợp lệ, không markdown và không text ngoài JSON. Mục tiêu của thí nghiệm này là đánh giá khả năng triển khai một LLM nhỏ cho sinh nội dung học tập có cấu trúc trong điều kiện tài nguyên hạn chế.

## 4. Dữ liệu và độ đo đánh giá

### 4.1. Dữ liệu Bi-Encoder

Tổng số cặp query-passage dùng trong notebook V2 là `4014`. Dữ liệu được chia theo `(source, chapter)`:


| Split      | Số mẫu |
| ---------- | ------ |
| Train      | 2787   |
| Validation | 616    |
| Test       | 611    |


Phân bố intent trong test set:


| Intent      | Số mẫu |
| ----------- | ------ |
| factual     | 191    |
| relational  | 212    |
| applicative | 208    |


### 4.2. Dữ liệu QLoRA Qwen

Dataset Qwen sạch, không flashcard có 1500 mẫu:


| Split      | Số mẫu |
| ---------- | ------ |
| Train      | 1200   |
| Validation | 150    |
| Test       | 150    |


Task distribution trong lần chạy mới:


| Split      | grounded_qa | question_generation | mcq_generation | summary |
| ---------- | ----------- | ------------------- | -------------- | ------- |
| Train      | 638         | 246                 | 159            | 157     |
| Validation | 64          | 32                  | 26             | 28      |
| Test       | 84          | 22                  | 19             | 25      |


### 4.3. Độ đo retrieval

Các độ đo retrieval được sử dụng:

- `Recall@1`: tỷ lệ câu hỏi có passage đúng ở vị trí đầu tiên.
- `Recall@5`: tỷ lệ câu hỏi có passage đúng trong top 5.
- `Recall@10`: tỷ lệ câu hỏi có passage đúng trong top 10.
- `MRR@10`: Mean Reciprocal Rank, đo thứ hạng của passage đúng đầu tiên trong top 10.

Các chỉ số này phù hợp với bài toán RAG vì LLM chỉ nhận một số lượng nhỏ context ở đầu danh sách. Nếu passage đúng nằm càng cao, khả năng sinh câu trả lời đúng càng cao.

### 4.4. Độ đo generation

Đối với QLoRA Qwen, các độ đo gồm:

- `validation loss`: loss trên validation set.
- `test loss`: loss trên test set độc lập.
- `json_valid_rate`: tỷ lệ output parse được JSON.
- `question_rougeL`: độ tương đồng bề mặt giữa câu hỏi sinh ra và target question.
- Rubric thủ công 50 mẫu test để đánh giá schema, độ đúng nguồn, không hallucination và chất lượng câu hỏi/câu trả lời.

## 5. Kết quả đánh giá

### 5.1. Kết quả fine-tuning Bi-Encoder trong notebook V2

Kết quả từ `Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb`:


| Model            | Val MRR@10 | Test MRR@10 | Delta test so với base |
| ---------------- | ---------- | ----------- | ---------------------- |
| Base             | 0.4234     | 0.4178      | +0.0000                |
| V1 MNRL          | 0.4658     | 0.4795      | +0.0617                |
| V2 Hard Negative | 0.4864     | 0.4982      | +0.0805                |


Kết quả này cho thấy fine-tuning giúp cải thiện retrieval so với base model. Hard negative mining tiếp tục cải thiện so với V1, với test MRR@10 tăng từ `0.4795` lên `0.4982`.

### 5.2. So sánh embedding trên cùng test split

Sau khi tải model V2 về local, đồ án chạy thêm script `scripts/compare_biencoder_v2_retrieval.py` để so sánh nhiều embedding trên cùng test split 611 queries. Kết quả được lưu tại:

- `artifacts/compare_biencoder_v2_retrieval.csv`
- `artifacts/compare_biencoder_v2_retrieval.json`


| Model                      | Recall@1   | Recall@5   | Recall@10  | MRR@10     |
| -------------------------- | ---------- | ---------- | ---------- | ---------- |
| Base Vietnamese Bi-Encoder | 0.6530     | 0.9051     | 0.9493     | 0.7603     |
| BGE-M3                     | **0.8052** | **0.9673** | **0.9902** | **0.8730** |
| E5-Large                   | 0.7971     | 0.9574     | 0.9869     | 0.8655     |
| V2 Hard Negative           | 0.7676     | 0.9525     | 0.9738     | 0.8443     |


So với base Vietnamese Bi-Encoder, V2 cải thiện rõ rệt:


| Metric    | Base   | V2     | Delta   |
| --------- | ------ | ------ | ------- |
| Recall@1  | 0.6530 | 0.7676 | +0.1146 |
| Recall@5  | 0.9051 | 0.9525 | +0.0475 |
| Recall@10 | 0.9493 | 0.9738 | +0.0245 |
| MRR@10    | 0.7603 | 0.8443 | +0.0840 |


Tương đối theo MRR@10, V2 cải thiện khoảng `+11.05%` so với base Vietnamese Bi-Encoder.

Tuy nhiên, V2 chưa vượt BGE-M3 và E5-Large trên benchmark này. BGE-M3 đạt MRR@10 cao nhất với `0.8730`, tiếp theo là E5-Large với `0.8655`, sau đó là V2 với `0.8443`. Điều này cho thấy fine-tuning giúp thu hẹp khoảng cách với các embedding mạnh, nhưng chưa vượt được các baseline lớn hơn hoặc tổng quát hơn.

### 5.3. Kết quả QLoRA Qwen2.5-1.5B-Instruct

Kết quả từ `Notebook/fine-tuning-qwen-new.ipynb`:


| Chỉ số                 | Giá trị |
| ---------------------- | ------- |
| Training loss cuối     | 0.0507  |
| Validation loss        | 0.1441  |
| Test loss              | 0.0807  |
| JSON valid rate        | 1.0000  |
| Question ROUGE-L       | 0.4922  |
| Số mẫu generation eval | 80      |


So với lần chạy trước:


| Chỉ số           | Lần trước | Lần mới |
| ---------------- | --------- | ------- |
| Validation loss  | 0.2861    | 0.1441  |
| JSON valid rate  | 0.6000    | 1.0000  |
| Question ROUGE-L | 0.6074    | 0.4922  |


Kết quả cho thấy việc làm sạch dữ liệu, bỏ flashcard và ràng buộc output JSON giúp cải thiện đáng kể độ ổn định định dạng. JSON valid rate tăng từ `0.6` lên `1.0`. ROUGE-L giảm so với lần trước, nhưng đây không nhất thiết là dấu hiệu xấu vì câu hỏi sinh ra có thể đúng nội dung nhưng khác cách diễn đạt so với target.

### 5.4. Nhận xét kết quả

Các kết quả retrieval cho thấy fine-tuning Vietnamese Bi-Encoder có hiệu quả rõ ràng khi so với base model cùng họ. Đây là đóng góp học sâu chính của đồ án và phù hợp trực tiếp với bài toán RAG Q&A, vì chất lượng retrieval quyết định chất lượng context được đưa vào LLM.

Tuy nhiên, so với BGE-M3 và E5-Large, mô hình V2 vẫn thấp hơn. Điều này là hợp lý vì BGE-M3 và E5-Large là các embedding model mạnh, được huấn luyện trên dữ liệu lớn và đa dạng. Kết quả này không làm mất ý nghĩa của fine-tuning, mà cho thấy fine-tuning giúp mô hình tiếng Việt chuyên biệt cải thiện đáng kể, đồng thời cung cấp một hướng nghiên cứu rõ ràng để tiếp tục mở rộng dữ liệu và cải thiện hard negative mining.

QLoRA Qwen là phần mở rộng cho generation có cấu trúc. Phần này hữu ích cho chức năng sinh câu hỏi/tóm tắt, nhưng trong phạm vi RAG Q&A, trọng tâm chính vẫn là fine-tuning retriever và đánh giá retrieval.

## 6. Những cải tiến và đóng góp học sâu

### 6.1. Fine-tuning retriever cho miền giáo trình

Ban đầu, hệ thống UniPolis chủ yếu sử dụng các mô hình embedding có sẵn. Điều này khiến đồ án có nguy cơ bị đánh giá là tích hợp công cụ hơn là áp dụng kỹ thuật học sâu. Để giải quyết vấn đề này, đồ án fine-tune Vietnamese Bi-Encoder trên các cặp query-document thuộc miền giáo trình lý luận chính trị.

Việc fine-tuning giúp mô hình học cách ánh xạ câu hỏi của sinh viên và đoạn giáo trình liên quan vào gần nhau hơn trong không gian vector. Đây là một cải tiến trực tiếp cho thành phần retrieval của RAG.

### 6.2. Hard negative mining

Sau vòng train V1, mô hình được dùng để tìm các hard negatives trong train set. Đây là các đoạn văn có vẻ liên quan theo embedding nhưng không phải positive đúng. Việc đưa các negative khó này vào vòng train V2 giúp mô hình học ranh giới phân biệt tốt hơn giữa các đoạn gần nghĩa.

Kết quả cho thấy V2 cải thiện so với V1 trên test set độc lập, chứng minh hard negative mining có tác dụng trong bài toán retrieval giáo trình.

### 6.3. Test set độc lập theo chapter-disjoint split

Một điểm quan trọng trong thực nghiệm là cách chia dữ liệu. Dataset có sẵn train/val/test, nhưng các split này vẫn overlap theo chapter. Nếu cùng một chương xuất hiện ở cả train và test, mô hình có thể được đánh giá lạc quan hơn vì đã thấy nội dung cùng miền hẹp trong train.

Notebook V2 tự split lại theo `(source, chapter)` để train, validation và test không trùng chapter key. Điều này giúp kết quả đánh giá đáng tin cậy hơn.

### 6.4. QLoRA LLM nhỏ cho generation có cấu trúc

Đồ án cũng thử nghiệm QLoRA Qwen2.5-1.5B-Instruct để sinh output dạng JSON cho các tác vụ học tập. Việc fine-tune LLM nhỏ không phải trọng tâm chính của RAG Q&A, nhưng là một đóng góp mở rộng cho chức năng tạo đề, sinh câu hỏi và tóm tắt.

Kết quả JSON valid rate đạt `1.0` cho thấy mô hình đã học tốt định dạng output có cấu trúc, phù hợp nếu hệ thống cần parse tự động kết quả sinh.

## 7. Kết luận và hướng phát triển

### 7.1. Kết quả đạt được

Đồ án UniPolis đã xây dựng được một hệ thống RAG hoàn chỉnh cho 6 giáo trình lý luận chính trị tiếng Việt. Hệ thống có khả năng xử lý PDF, xây dựng vector store và BM25 index, truy hồi context, sinh câu trả lời dựa trên ngữ cảnh và hiển thị nguồn tham chiếu qua giao diện Streamlit.

Về mặt học sâu, đồ án đã fine-tune Vietnamese Bi-Encoder bằng contrastive learning và hard negative mining. Mô hình V2 cải thiện `MRR@10` từ `0.7603` lên `0.8443` so với base Vietnamese Bi-Encoder trên cùng test split, tương ứng mức tăng khoảng `+11.05%`. Khi so với các baseline mạnh hơn, V2 chưa vượt BGE-M3 và E5-Large, nhưng đã cho thấy hiệu quả rõ ràng của fine-tuning trong cùng họ mô hình.

Ngoài ra, đồ án đã thử nghiệm QLoRA Qwen2.5-1.5B-Instruct cho sinh nội dung học tập có cấu trúc, đạt JSON valid rate `1.0` trên quick generation evaluation.

### 7.2. Hạn chế

Đồ án vẫn còn một số hạn chế:

- Dữ liệu PDF có nhiễu, lỗi xuống dòng và một số chunk bị đứt câu.
- Fine-tuning Bi-Encoder mới dùng dữ liệu của 3 môn có `training_data_`*, chưa bao phủ đủ 6 giáo trình.
- Mô hình V2 cải thiện so với base Vietnamese Bi-Encoder nhưng chưa vượt BGE-M3 và E5-Large.
- Đánh giá end-to-end RAG Q&A bằng RAGAS hoặc human evaluation chưa được thực hiện đầy đủ.
- QLoRA Qwen mới là thí nghiệm generation có cấu trúc, chưa phải thành phần chính được tích hợp hoàn toàn vào app RAG.

### 7.3. Hướng phát triển

Các hướng phát triển tiếp theo gồm:

- Mở rộng dữ liệu fine-tuning Bi-Encoder cho đủ 6 giáo trình.
- Xây dựng bộ câu hỏi kiểm thử chuẩn cho từng môn với gold context rõ ràng.
- Đánh giá end-to-end RAG bằng Faithfulness, Context Precision, Context Recall và Answer Correctness.
- Tích hợp đầy đủ vector store build bằng Vietnamese Bi-Encoder V2 vào app và so sánh trực tiếp trên giao diện.
- Cải thiện metadata theo chương, mục, tiểu mục thay vì chỉ trang/chunk.
- Chấm thủ công 50 mẫu Qwen theo rubric để đánh giá chất lượng generation.

## 8. Bảng phân công

Nếu làm nhóm, có thể dùng bảng sau:


| Nhiệm vụ                           | Thành viên 1 | Thành viên 2 | Thành viên 3 | Thành viên 4 |
| ---------------------------------- | ------------ | ------------ | ------------ | ------------ |
| Tìm hiểu RAG và tài liệu liên quan | X            | X            | X            | X            |
| Xây dựng ingest/RAG engine         | X            |              |              |              |
| Giao diện Streamlit                |              | X            |              |              |
| Fine-tuning Bi-Encoder             |              |              | X            |              |
| QLoRA Qwen và instruction data     |              |              |              | X            |
| Thực nghiệm và đánh giá            | X            | X            | X            | X            |
| Viết báo cáo/slide                 | X            | X            | X            | X            |


Nếu làm cá nhân, thay bằng bảng nhiệm vụ:


| Nhiệm vụ                                | Trạng thái       |
| --------------------------------------- | ---------------- |
| Xây dựng pipeline RAG                   | Hoàn thành       |
| Xây dựng giao diện Streamlit            | Hoàn thành       |
| Fine-tuning Vietnamese Bi-Encoder       | Hoàn thành       |
| Hard negative mining và test evaluation | Hoàn thành       |
| QLoRA Qwen instruction tuning           | Hoàn thành       |
| Đánh giá retrieval nhiều embedding      | Hoàn thành       |
| Đánh giá end-to-end RAG bằng RAGAS      | Hướng phát triển |


## 9. Tài liệu tham khảo

1. Patrick Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.
2. Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP, 2019.
3. Vladimir Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020.
4. Edward J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.
5. Tim Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS, 2023.
6. Qwen Team. Qwen2.5 Technical Report / Model Card.
7. ChromaDB Documentation.
8. Sentence-Transformers Documentation.
9. LangChain Documentation.
10. BAAI. BGE-M3 and BGE Reranker model cards.

## 10. Phụ lục

### Phụ lục A - File và thư mục quan trọng


| File/thư mục                                                                  | Vai trò                             |
| ----------------------------------------------------------------------------- | ----------------------------------- |
| `app.py`                                                                      | Entry point Streamlit               |
| `rag_engine.py`                                                               | Lõi RAG retrieval/generation        |
| `config.py`                                                                   | Cấu hình môn học và pipeline mode   |
| `ingest.py`                                                                   | Ingest lõi                          |
| `ingest_political_6_recursive_bge.py`                                         | Ingest 6 giáo trình                 |
| `exam_generator.py`, `exam_tab.py`                                            | Tạo đề/câu hỏi                      |
| `Notebook/fine-tuning-vietnamese-bi-encoder-v2.ipynb`                         | Fine-tuning Bi-Encoder V2           |
| `scripts/compare_biencoder_v2_retrieval.py`                                   | Benchmark embedding trên test split |
| `artifacts/compare_biencoder_v2_retrieval.csv`                                | Kết quả so sánh embedding           |
| `Notebook/fine-tuning-qwen-new.ipynb`                                         | QLoRA Qwen                          |
| `scripts/generate_qwen_instruction_data.py`                                   | Sinh instruction data Qwen          |
| `data/qwen_lora_instruction_data_clean_no_flashcard/`                         | Dataset Qwen chính                  |
| `bi-encoder-finetuned/models/bi_encoder_hnm_v2/vietnamese-bi-encoder-v2-hnm/` | Model Bi-Encoder V2 tải từ Kaggle   |


### Phụ lục B - Checklist hình và bảng nên đưa vào bản PDF

- Hình kiến trúc UniPolis tổng thể.
- Hình offline ingest pipeline.
- Hình online RAG pipeline.
- Bảng 6 giáo trình.
- Bảng pipeline mode.
- Bảng split dữ liệu Bi-Encoder.
- Bảng kết quả base/V1/V2 từ notebook.
- Bảng so sánh base VN/BGE-M3/E5-Large/V2.
- Bảng kết quả QLoRA Qwen.
- Ảnh demo Streamlit hỏi đáp.
- Ảnh hiển thị nguồn tham chiếu.
- Ảnh chức năng tạo đề/câu hỏi nếu có.

