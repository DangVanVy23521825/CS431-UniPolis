# TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN

# KHOA KHOA HỌC MÁY TÍNH

# BÁO CÁO ĐỒ ÁN CUỐI KỲ

## Môn học: Các Kỹ thuật học sâu và ứng dụng - CS431

## Đề tài: Cải thiện truy xuất ngữ nghĩa tiếng Việt bằng MultipleNegativesRankingLoss và Hard Negative Mining cho hệ thống RAG giáo trình

### Giảng viên hướng dẫn

TS. ........................................

### Nhóm thực hiện


| STT | Họ và tên                                | MSSV     |
| --- | ---------------------------------------- | -------- |
| 1   | ........................................ | ........ |
| 2   | ........................................ | ........ |
| 3   | ........................................ | ........ |
| 4   | ........................................ | ........ |


TP. Hồ Chí Minh, 2026

---

## Mục lục


| Mục | Nội dung                                    |
| --- | ------------------------------------------- |
| 1   | Giới thiệu đề tài                           |
| 2   | Cơ sở lý thuyết                             |
| 3   | Phương pháp đề xuất                         |
| 4   | Dữ liệu và độ đo đánh giá                   |
| 5   | Thực nghiệm truy xuất                       |
| 6   | Ứng dụng RAG chatbot và đánh giá downstream |
| 7   | Kết luận                                    |
| 8   | Bảng phân công                              |
| 9   | Tài liệu tham khảo                          |


## Danh mục bảng


| Bảng | Nội dung                                                        |
| ---- | --------------------------------------------------------------- |
| 1    | Số lượng mẫu trong từng split của dữ liệu fine-tuning retriever |
| 2    | Phân bố loại câu hỏi trong tập test                             |
| 3    | Kết quả fine-tuning Vietnamese Bi-Encoder theo hai giai đoạn    |
| 4    | So sánh các mô hình embedding trên cùng test split              |
| 5    | Cấu hình fine-tuning retriever                                  |
| 6    | Các tiêu chí RAGAS dùng để đánh giá chatbot downstream          |


---

## Tóm tắt

Truy xuất ngữ nghĩa là một thành phần quan trọng trong các hệ thống hỏi đáp tài liệu dài. Đối với các giáo trình tiếng Việt có nội dung học thuật, người học thường đặt câu hỏi bằng ngôn ngữ tự nhiên, trong khi thông tin cần tìm có thể được diễn đạt bằng thuật ngữ hoặc cấu trúc câu khác trong tài liệu gốc. Các phương pháp tìm kiếm dựa trên từ khóa có thể bỏ sót những trường hợp tương đồng về ý nghĩa nhưng không trùng khớp bề mặt. Ngược lại, các mô hình embedding tổng quát tuy có khả năng biểu diễn ngữ nghĩa tốt nhưng không phải lúc nào cũng tối ưu cho miền dữ liệu chuyên biệt như giáo trình lý luận chính trị tiếng Việt.

Trong đồ án này, chúng em tập trung vào bài toán fine-tuning mô hình Bi-Encoder cho truy xuất ngữ nghĩa tiếng Việt. Thành phần học sâu chính của đề tài là huấn luyện retriever bằng contrastive learning với `MultipleNegativesRankingLoss`, sau đó cải thiện thêm bằng Hard Negative Mining. Mục tiêu của quá trình huấn luyện là đưa vector của câu hỏi và đoạn văn đúng lại gần nhau trong không gian embedding, đồng thời đẩy xa các đoạn không liên quan. Với Hard Negative Mining, mô hình được huấn luyện thêm trên các đoạn sai nhưng có độ tương đồng cao, qua đó học cách phân biệt các ngữ cảnh gần chủ đề nhưng không trả lời đúng câu hỏi.

Dữ liệu thực nghiệm gồm 4014 cặp câu hỏi và đoạn văn liên quan được xây dựng từ các giáo trình lý luận chính trị tiếng Việt. Dữ liệu được chia thành train, validation và test theo đơn vị `(source, chapter)` để hạn chế rò rỉ dữ liệu giữa các tập. Kết quả thực nghiệm cho thấy Vietnamese Bi-Encoder sau khi fine-tune bằng `MultipleNegativesRankingLoss` cải thiện MRR@10 trên test set từ 0.4178 lên 0.4795. Khi bổ sung Hard Negative Mining, MRR@10 tiếp tục tăng lên 0.4982. Trong một thiết lập benchmark khác trên cùng test split 611 truy vấn, mô hình fine-tuned đạt MRR@10 là 0.8443, cải thiện 11.05% so với Vietnamese Bi-Encoder gốc.

Để minh họa giá trị ứng dụng của retriever đã fine-tune, đồ án xây dựng hệ thống UniPolis, một chatbot hỏi đáp giáo trình dựa trên Retrieval-Augmented Generation. Trong hệ thống này, RAG không phải đóng góp kỹ thuật chính mà là ứng dụng downstream dùng để đánh giá tác động thực tế của retriever. Chatbot sẽ được đánh giá bằng RAGAS thông qua các tiêu chí như Faithfulness, Answer Relevancy, Context Precision và Context Recall. Cách tiếp cận này giúp đồ án có hai tầng đánh giá: đánh giá nội tại chất lượng truy xuất bằng MRR/Recall và đánh giá ứng dụng đầu cuối bằng RAGAS.

## 1. Giới thiệu đề tài

### 1.1. Bối cảnh

Trong các hệ thống hỏi đáp tài liệu dài, mô hình không thể xử lý toàn bộ kho tri thức trong một lần suy luận. Vì vậy, hệ thống cần một cơ chế truy xuất để chọn ra những đoạn ngữ cảnh liên quan nhất với câu hỏi. Chất lượng truy xuất có ảnh hưởng trực tiếp đến chất lượng câu trả lời cuối cùng. Nếu retriever không tìm được đúng đoạn chứa thông tin cần thiết, mô hình sinh phía sau khó có thể trả lời chính xác dù có năng lực ngôn ngữ mạnh.

Với tiếng Việt, đặc biệt là văn bản học thuật, bài toán truy xuất ngữ nghĩa gặp nhiều khó khăn. Cùng một nội dung có thể được diễn đạt theo nhiều cách khác nhau, trong khi các khái niệm chuyên ngành thường có quan hệ gần nghĩa, bao hàm hoặc đối lập. Ví dụ, một câu hỏi về “vai trò lịch sử” có thể liên quan đến đoạn văn dùng các cụm như “ý nghĩa”, “tác động”, “giá trị” hoặc “đóng góp”. Tìm kiếm từ khóa không đủ linh hoạt trong các trường hợp này.

Dense retrieval giải quyết vấn đề trên bằng cách ánh xạ câu hỏi và đoạn văn vào cùng một không gian vector. Trong không gian này, những văn bản có ý nghĩa gần nhau sẽ có biểu diễn gần nhau. Tuy nhiên, mô hình embedding tổng quát thường được huấn luyện trên dữ liệu đa miền, nên có thể chưa tối ưu cho miền giáo trình lý luận chính trị. Do đó, fine-tuning retriever trên dữ liệu miền hẹp là một hướng phù hợp để cải thiện chất lượng truy xuất.

### 1.2. Vấn đề nghiên cứu

Đồ án đặt trọng tâm vào câu hỏi: làm thế nào để cải thiện mô hình truy xuất ngữ nghĩa tiếng Việt cho miền giáo trình bằng các kỹ thuật học sâu? Cụ thể, đồ án nghiên cứu việc fine-tune Bi-Encoder bằng contrastive learning, trong đó mỗi câu hỏi được ghép với một đoạn văn đúng và mô hình học cách xếp hạng đoạn đúng cao hơn các đoạn sai.

Bài toán được mô tả như sau. Cho tập các cặp huấn luyện `(q, p+)`, trong đó `q` là câu hỏi và `p+` là đoạn văn liên quan. Mục tiêu là học hàm mã hóa `f(.)` sao cho độ tương đồng giữa `f(q)` và `f(p+)` cao hơn độ tương đồng giữa `f(q)` và các đoạn không liên quan. Khi có truy vấn mới, hệ thống mã hóa truy vấn thành vector và tìm các đoạn văn có vector gần nhất trong kho tài liệu.

### 1.3. Động lực chọn MultipleNegativesRankingLoss và Hard Negative Mining

`MultipleNegativesRankingLoss` phù hợp với bài toán truy xuất vì nó tận dụng các mẫu trong cùng một batch để tạo negative tự nhiên. Với một batch gồm nhiều cặp query-positive, positive của các query khác được xem như negative cho query hiện tại. Cách này giúp mô hình học xếp hạng tương đối mà không cần chuẩn bị thủ công nhiều negative cho từng truy vấn.

Tuy nhiên, negative trong batch thường chưa đủ khó. Nhiều đoạn sai có nội dung khác xa đoạn đúng nên mô hình dễ phân biệt. Trong thực tế truy xuất tài liệu, lỗi thường xảy ra khi mô hình nhầm giữa các đoạn cùng chương, cùng chủ đề hoặc cùng thuật ngữ nhưng không trả lời đúng câu hỏi. Hard Negative Mining giải quyết vấn đề này bằng cách tìm các đoạn sai có điểm tương đồng cao với câu hỏi, sau đó đưa chúng vào giai đoạn huấn luyện tiếp theo. Nhờ đó, mô hình học được ranh giới phân biệt tinh tế hơn giữa positive và các đoạn gần nghĩa.

### 1.4. Đóng góp chính

Đóng góp chính của đồ án là xây dựng và đánh giá quy trình fine-tuning Bi-Encoder cho truy xuất ngữ nghĩa tiếng Việt trên miền giáo trình lý luận chính trị. Quy trình gồm hai giai đoạn: huấn luyện ban đầu bằng `MultipleNegativesRankingLoss` và huấn luyện tăng cường bằng Hard Negative Mining.

Đồ án xây dựng tập dữ liệu retrieval gồm 4014 cặp query-passage và chia dữ liệu theo chương để hạn chế rò rỉ ngữ cảnh giữa train, validation và test. Đây là một yếu tố quan trọng vì nếu cùng một chương xuất hiện ở cả train và test, kết quả đánh giá có thể cao hơn thực tế.

Đồ án đánh giá mô hình bằng các độ đo truy xuất tiêu chuẩn như MRR@10 và Recall@k. Ngoài ra, đồ án so sánh mô hình fine-tuned với các embedding model mạnh như BGE-M3 và multilingual-E5-Large để xác định vị trí của mô hình trong bối cảnh các retriever hiện đại.

Cuối cùng, đồ án triển khai UniPolis như một ứng dụng downstream theo kiến trúc RAG. Ứng dụng này dùng retriever đã fine-tune để truy xuất ngữ cảnh cho chatbot hỏi đáp giáo trình và được đánh giá bằng RAGAS nhằm kiểm tra tác động của retrieval đến chất lượng trả lời đầu cuối.

## 2. Cơ sở lý thuyết

### 2.1. Bài toán truy xuất thông tin

Truy xuất thông tin có thể được mô hình hóa như một bài toán xếp hạng. Cho một truy vấn `q` và một tập tài liệu hoặc đoạn văn `D = {p1, p2, ..., pn}`, hệ thống cần tìm một hàm scoring `s(q, p)` để gán điểm liên quan giữa truy vấn và từng đoạn văn. Kết quả truy xuất là danh sách các đoạn được sắp xếp giảm dần theo điểm:

```text
rank(q, D) = argsort_{p in D} s(q, p)
```

Trong bài toán hỏi đáp tài liệu, đơn vị truy xuất thường không phải toàn bộ tài liệu mà là các đoạn ngữ cảnh ngắn hơn. Nếu đoạn quá dài, embedding có thể bị nhiễu bởi nhiều ý khác nhau; nếu đoạn quá ngắn, thông tin trả lời có thể bị thiếu. Vì vậy, trước khi truy xuất, tài liệu thường được chia thành các passage có độ dài cố định hoặc gần cố định.

Về mặt học máy, mục tiêu của retriever là đưa đoạn đúng `p+` lên vị trí càng cao càng tốt trong danh sách kết quả. Với một truy vấn `q`, nếu `p+` là đoạn chứa thông tin cần thiết và `p-` là đoạn không liên quan hoặc không trả lời đúng, mục tiêu lý tưởng là:

```text
s(q, p+) > s(q, p-)
```

Điều này biến retrieval thành một bài toán learning-to-rank, trong đó mô hình không chỉ học phân loại đúng/sai mà còn học thứ tự ưu tiên giữa các đoạn văn.

### 2.2. Sparse Retrieval và Dense Retrieval

Các phương pháp truy xuất truyền thống như TF-IDF hoặc BM25 thuộc nhóm sparse retrieval. Chúng biểu diễn văn bản bằng vector thưa dựa trên từ vựng và tính điểm liên quan dựa trên mức độ trùng khớp từ khóa. BM25 thường được mô tả bởi công thức:

```text
BM25(q, p) = sum_{t in q} IDF(t) * ((f(t, p) * (k1 + 1)) / (f(t, p) + k1 * (1 - b + b * |p| / avgdl)))
```

Trong đó `f(t, p)` là tần suất của từ `t` trong đoạn `p`, `|p|` là độ dài đoạn, `avgdl` là độ dài trung bình của các đoạn, còn `k1` và `b` là các siêu tham số. BM25 hiệu quả khi truy vấn và văn bản có nhiều từ khóa trùng nhau, nhưng gặp hạn chế khi câu hỏi được diễn đạt khác với tài liệu gốc.

Dense Retrieval khắc phục hạn chế này bằng cách biểu diễn truy vấn và đoạn văn dưới dạng vector đặc trong không gian liên tục. Thay vì dựa vào trùng khớp bề mặt, mô hình neural network học biểu diễn ngữ nghĩa của văn bản. Nếu gọi encoder là `f_theta(.)`, ta có:

```text
u = f_theta(q)
v = f_theta(p)
```

Trong đó `u` là embedding của truy vấn và `v` là embedding của đoạn văn. Điểm liên quan có thể được tính bằng cosine similarity:

```text
s(q, p) = cos(u, v) = (u · v) / (||u|| ||v||)
```

Nếu các vector đã được chuẩn hóa về độ dài 1, cosine similarity tương đương với dot product:

```text
s(q, p) = u · v
```

Ưu điểm của Dense Retrieval là khả năng tìm kiếm theo ngữ nghĩa. Hai văn bản có thể ít trùng từ nhưng vẫn có vector gần nhau nếu chúng biểu đạt cùng một ý. Đây là lý do Dense Retrieval phù hợp với hệ thống hỏi đáp bằng ngôn ngữ tự nhiên.

### 2.3. Bi-Encoder và Cross-Encoder

Trong retrieval hiện đại, hai kiến trúc phổ biến là Bi-Encoder và Cross-Encoder. Bi-Encoder mã hóa truy vấn và đoạn văn độc lập:

```text
u = f_theta(q)
v = f_theta(p)
s(q, p) = sim(u, v)
```

Do đoạn văn được mã hóa độc lập, toàn bộ corpus có thể được encode trước và lưu trong vector database. Khi có truy vấn mới, hệ thống chỉ cần encode truy vấn và tìm kiếm nearest neighbors. Độ phức tạp suy luận khi truy vấn giảm đáng kể vì embedding của các đoạn đã được tính trước.

Cross-Encoder xử lý truy vấn và đoạn văn cùng lúc bằng cách đưa cặp `(q, p)` vào mô hình:

```text
s(q, p) = g_theta([q; p])
```

Cross-Encoder thường cho điểm chính xác hơn vì mô hình có thể quan sát tương tác token-level giữa truy vấn và đoạn văn. Tuy nhiên, nó phải chạy mô hình cho từng cặp `(q, p)`, nên không phù hợp để tìm kiếm trực tiếp trên corpus lớn. Trong thực tế, Bi-Encoder thường được dùng cho bước retrieval ban đầu, còn Cross-Encoder có thể dùng cho reranking trên một số ít ứng viên top-k.

Đồ án tập trung vào Bi-Encoder vì đây là thành phần có thể huấn luyện và triển khai hiệu quả trong pipeline hỏi đáp. Chất lượng embedding của Bi-Encoder quyết định khả năng đưa đoạn đúng vào top-k context cho hệ thống RAG.

### 2.4. Sentence Transformer và pooling embedding

Sentence Transformer là họ mô hình học biểu diễn câu hoặc đoạn văn dựa trên Transformer encoder. Đầu vào là một chuỗi token `x = (x1, x2, ..., xm)`. Transformer tạo ra hidden states tương ứng:

```text
H = Transformer(x) = (h1, h2, ..., hm)
```

Để thu được một vector duy nhất cho toàn bộ câu hoặc đoạn văn, mô hình cần một phép pooling. Một lựa chọn phổ biến là mean pooling trên các token hợp lệ:

```text
z = (1 / M) * sum_{i=1}^{M} hi
```

Sau đó, vector có thể được chuẩn hóa L2:

```text
e = z / ||z||
```

Việc chuẩn hóa embedding giúp dot product tương đương cosine similarity và ổn định hơn khi so sánh vector. Trong retrieval, điều này quan trọng vì mô hình cần xếp hạng hàng nghìn hoặc hàng triệu đoạn văn theo cùng một thước đo similarity.

### 2.5. Contrastive Learning cho retrieval

Contrastive learning là phương pháp học biểu diễn dựa trên quan hệ tương đồng và không tương đồng giữa các mẫu. Với retrieval, mỗi mẫu huấn luyện thường gồm một query `q`, một positive passage `p+` và một hoặc nhiều negative passages `p-`.

Mục tiêu của contrastive learning là làm cho embedding của query gần embedding của positive hơn negative. Một dạng mục tiêu theo margin có thể viết là:

```text
L_margin = max(0, gamma - s(q, p+) + s(q, p-))
```

Trong đó `gamma` là margin. Loss bằng 0 khi điểm của positive lớn hơn điểm của negative ít nhất `gamma`. Tuy nhiên, trong các hệ thống Sentence Transformer, dạng softmax contrastive loss thường được sử dụng nhiều hơn vì nó tận dụng được nhiều negative trong cùng một batch.

Về bản chất, contrastive learning định hình không gian embedding sao cho các điểm cùng ngữ nghĩa tạo thành các cụm gần nhau, còn các điểm khác ngữ nghĩa bị đẩy xa nhau. Đối với giáo trình lý luận chính trị, điều này giúp mô hình học được rằng câu hỏi và đoạn trả lời đúng có thể không trùng từ nhưng vẫn phải gần nhau về mặt ngữ nghĩa.

### 2.6. In-batch Negatives

Một khó khăn trong huấn luyện retrieval là cần nhiều negative để mô hình học xếp hạng tốt. Nếu phải gán nhãn thủ công negative cho từng query, chi phí xây dựng dữ liệu sẽ rất lớn. In-batch negatives là kỹ thuật giải quyết vấn đề này bằng cách tận dụng các positive của mẫu khác trong cùng batch làm negative.

Giả sử một batch có `N` cặp query-positive:

```text
B = {(q1, p1+), (q2, p2+), ..., (qN, pN+)}
```

Với query `qi`, passage `pi+` là positive. Các passage còn lại trong batch `{pj+ | j != i}` được xem là negative. Khi đó, chỉ với `N` cặp được gán nhãn positive, mỗi query có thêm `N - 1` negative mà không cần gán nhãn riêng.

Ma trận similarity của batch có kích thước `N x N`:

```text
S_ij = s(qi, pj+)
```

Đường chéo chính của ma trận là điểm của các cặp đúng, còn các phần tử ngoài đường chéo là điểm của in-batch negatives. Huấn luyện sẽ khuyến khích các phần tử đường chéo lớn hơn các phần tử ngoài đường chéo theo từng hàng.

### 2.7. MultipleNegativesRankingLoss

`MultipleNegativesRankingLoss` là loss function dựa trên softmax contrastive learning. Với batch gồm `N` cặp `(qi, pi+)`, xác suất mô hình chọn đúng positive cho query `qi` được tính như sau:

```text
P(pi+ | qi) = exp(s(qi, pi+) / tau) / sum_{j=1}^{N} exp(s(qi, pj+) / tau)
```

Trong đó `tau` là temperature. Temperature điều chỉnh độ sắc của phân phối softmax. Khi `tau` nhỏ, mô hình phạt mạnh hơn các negative có điểm gần positive; khi `tau` lớn, phân phối trở nên mềm hơn.

Loss trên toàn batch là negative log-likelihood của các positive đúng:

```text
L_MNRL = -(1 / N) * sum_{i=1}^{N} log P(pi+ | qi)
```

Viết dưới dạng ma trận similarity `S`, loss tương đương với cross-entropy theo từng hàng, trong đó nhãn đúng của hàng `i` là cột `i`:

```text
L_MNRL = CrossEntropy(S / tau, y)
y_i = i
```

Điểm mạnh của `MultipleNegativesRankingLoss` là nó tối ưu trực tiếp cho bài toán ranking. Mô hình không chỉ học rằng cặp `(qi, pi+)` là đúng, mà còn học rằng `pi+` phải được xếp cao hơn các passage khác trong batch. Điều này phù hợp với mục tiêu retrieval, nơi thứ hạng của đoạn đúng quan trọng hơn xác suất phân loại độc lập.

Tuy nhiên, chất lượng của loss phụ thuộc vào chất lượng negative. Nếu các in-batch negatives quá khác biệt với query, bài toán trở nên dễ và mô hình có thể không học được các ranh giới tinh tế. Vì vậy, Hard Negative Mining thường được kết hợp với MNRL để tăng độ khó của quá trình huấn luyện.

### 2.8. Hard Negative Mining

Negative có thể được chia thành easy negative và hard negative. Easy negative là đoạn sai có nội dung khác xa query, nên mô hình dễ dàng cho điểm thấp. Hard negative là đoạn sai nhưng có nội dung gần query, thường cùng chủ đề hoặc chứa các thuật ngữ tương tự. Đây là loại negative quan trọng hơn cho việc cải thiện retriever.

Về mặt toán học, với một query `q`, positive `p+` và tập corpus `D`, hard negative có thể được chọn theo công thức:

```text
p_hard- = argmax_{p in D, p != p+} s(q, p)
```

Trong thực tế, để giảm chi phí, hệ thống không cần duyệt toàn bộ corpus mà lấy top-k kết quả từ mô hình hiện tại, sau đó loại bỏ positive và chọn negative có điểm cao nhất:

```text
H(q) = topK_{p in D} s(q, p)
p_hard- = first p in H(q) such that p != p+
```

Sau khi khai thác hard negatives, dữ liệu huấn luyện có thể được biểu diễn dưới dạng triplet `(q, p+, p_hard-)`. Mục tiêu trực giác là:

```text
s(q, p+) > s(q, p_hard-)
```

Trong đồ án, hard negatives được dùng để tạo giai đoạn huấn luyện thứ hai. Dù loss chính vẫn là `MultipleNegativesRankingLoss`, phân phối dữ liệu đã thay đổi vì mô hình được tiếp xúc với các đoạn sai khó hơn. Điều này giúp mô hình cải thiện khả năng phân biệt giữa “đoạn cùng chủ đề” và “đoạn thật sự trả lời đúng câu hỏi”.

Hard Negative Mining cũng có rủi ro. Nếu dữ liệu gán nhãn chưa đầy đủ, một hard negative có thể thực chất cũng chứa thông tin đúng nhưng không được gán nhãn là positive. Trường hợp này gọi là false negative. Vì vậy, quá trình mining cần được thiết kế cẩn thận, đặc biệt trong các tài liệu mà cùng một nội dung có thể xuất hiện ở nhiều đoạn khác nhau.

### 2.9. Các độ đo đánh giá retrieval

Chất lượng retrieval thường được đánh giá bằng các độ đo dựa trên thứ hạng. Trong đồ án, hai nhóm độ đo chính là Recall@k và MRR@k.

Recall@k đo tỷ lệ truy vấn mà ít nhất một đoạn đúng xuất hiện trong top-k kết quả:

```text
Recall@k = (1 / |Q|) * sum_{q in Q} I(rank_q(p+) <= k)
```

Trong đó `I(.)` là hàm chỉ thị, bằng 1 nếu điều kiện đúng và bằng 0 nếu sai. Recall@1 phản ánh khả năng đưa đoạn đúng lên đầu, còn Recall@5 và Recall@10 phản ánh khả năng đưa đoạn đúng vào tập context ứng viên.

Mean Reciprocal Rank tại k đo vị trí của đoạn đúng đầu tiên trong top-k:

```text
MRR@k = (1 / |Q|) * sum_{q in Q} 1 / rank_q
```

Nếu đoạn đúng không xuất hiện trong top-k, reciprocal rank được tính bằng 0. MRR@k nhạy với vị trí của kết quả đúng, nên phù hợp với RAG vì mô hình sinh thường ưu tiên các đoạn đứng đầu danh sách context.

### 2.10. Retrieval-Augmented Generation như ứng dụng downstream

Retrieval-Augmented Generation kết hợp retriever và generator. Với truy vấn `q`, retriever chọn tập ngữ cảnh `Ck(q)` gồm top-k passage:

```text
Ck(q) = topK_{p in D} s(q, p)
```

Generator sinh câu trả lời `a` dựa trên truy vấn và các ngữ cảnh được truy xuất:

```text
a = G(q, Ck(q))
```

Trong kiến trúc này, retriever quyết định thông tin nào được đưa vào generator. Nếu `Ck(q)` không chứa đoạn đúng, generator có thể trả lời thiếu căn cứ hoặc suy diễn ngoài tài liệu. Vì vậy, cải thiện retriever là một hướng trực tiếp để cải thiện RAG.

Trong đồ án, RAG được xem là ứng dụng downstream của mô hình truy xuất đã fine-tune. Cách định vị này giúp tách rõ đóng góp kỹ thuật và ứng dụng. Đóng góp kỹ thuật nằm ở việc huấn luyện retriever bằng `MultipleNegativesRankingLoss` và Hard Negative Mining; chatbot RAG là môi trường để kiểm tra retriever trong một hệ thống hỏi đáp thực tế.

### 2.11. Đánh giá RAG bằng RAGAS

RAGAS là bộ tiêu chí đánh giá hệ thống Retrieval-Augmented Generation. Khác với các độ đo retrieval chỉ xét thứ hạng đoạn văn, RAGAS đánh giá cả ngữ cảnh được truy xuất và câu trả lời sinh ra.

Faithfulness đo mức độ các phát biểu trong câu trả lời được hỗ trợ bởi context. Nếu câu trả lời chứa thông tin không có trong ngữ cảnh, điểm faithfulness sẽ thấp. Answer Relevancy đo mức độ câu trả lời phù hợp với câu hỏi. Context Precision đánh giá tỷ lệ các đoạn context được truy xuất thật sự liên quan, còn Context Recall đánh giá mức độ context bao phủ thông tin cần thiết.

Trong mối liên hệ với retriever, Context Precision và Context Recall là hai tiêu chí phản ánh trực tiếp chất lượng truy xuất trong pipeline RAG. Nếu fine-tuning retriever làm tăng khả năng đưa đoạn đúng vào top-k, ta kỳ vọng Context Recall tăng. Nếu retriever xếp hạng các đoạn ít nhiễu hơn, Context Precision cũng có thể tăng. Faithfulness và Answer Relevancy cho biết cải thiện context có chuyển hóa thành câu trả lời tốt hơn hay không.

## 3. Phương pháp đề xuất

### 3.1. Tổng quan quy trình

Quy trình đề xuất gồm hai phần chính. Phần thứ nhất là fine-tuning retriever. Dữ liệu câu hỏi và đoạn văn đúng được dùng để huấn luyện Bi-Encoder bằng `MultipleNegativesRankingLoss`. Sau đó, mô hình vòng 1 được dùng để khai thác hard negatives và tiếp tục huấn luyện vòng 2.

Phần thứ hai là tích hợp retriever vào hệ thống RAG. Các giáo trình được xử lý, chia đoạn và lập chỉ mục vector. Khi người dùng đặt câu hỏi, retriever tìm các đoạn phù hợp nhất; các đoạn này được đưa vào mô hình ngôn ngữ để sinh câu trả lời có căn cứ.

### 3.2. Xây dựng dữ liệu query-passage

Dữ liệu huấn luyện retrieval gồm các cặp `(query, positive passage)`. `query` là câu hỏi tự nhiên về nội dung giáo trình, còn `positive passage` là đoạn văn chứa thông tin trả lời. Các cặp này được dùng làm tín hiệu giám sát để mô hình học mối quan hệ ngữ nghĩa giữa câu hỏi và ngữ cảnh.

Để hạn chế rò rỉ dữ liệu, quá trình chia tập không thực hiện ngẫu nhiên theo từng mẫu riêng lẻ. Thay vào đó, dữ liệu được nhóm theo nguồn và chương. Các chương được phân bổ vào train, validation và test sao cho cùng một chương không xuất hiện ở nhiều split. Cách chia này làm bài toán đánh giá khó hơn nhưng phản ánh tốt hơn khả năng tổng quát hóa sang phần nội dung chưa thấy trong huấn luyện.

### 3.3. Huấn luyện vòng 1 với MultipleNegativesRankingLoss

Ở vòng huấn luyện đầu tiên, mô hình Vietnamese Bi-Encoder được fine-tune bằng `MultipleNegativesRankingLoss`. Mỗi batch gồm nhiều cặp query-positive. Với mỗi query, positive tương ứng là đáp án đúng, còn positive của các query khác trong batch được xem là in-batch negatives.

Mục tiêu của vòng 1 là điều chỉnh không gian embedding để câu hỏi gần hơn với đoạn đúng. Sau vòng này, mô hình thường cải thiện đáng kể so với mô hình gốc vì đã học được đặc trưng của miền giáo trình và cách diễn đạt câu hỏi tiếng Việt trong tập dữ liệu.

### 3.4. Khai thác hard negatives

Sau vòng 1, mô hình được dùng để truy xuất top-k đoạn văn trong corpus train cho từng câu hỏi. Trong danh sách này, đoạn positive gốc được loại bỏ. Đoạn sai có điểm similarity cao nhất được chọn làm hard negative.

Hard negative có giá trị huấn luyện cao vì nó phản ánh đúng loại lỗi mà retriever có thể gặp khi vận hành. Nếu một đoạn sai vẫn được mô hình xếp hạng cao, nghĩa là embedding hiện tại chưa đủ phân biệt giữa đoạn đúng và đoạn gần chủ đề. Việc đưa đoạn này vào huấn luyện giúp mô hình cập nhật ranh giới quyết định tốt hơn.

### 3.5. Huấn luyện vòng 2 với hard negatives

Ở vòng 2, mô hình tiếp tục được huấn luyện từ checkpoint vòng 1. Dữ liệu huấn luyện lúc này vẫn giữ query-positive nhưng bổ sung negative khó đã khai thác. Mục tiêu là làm tăng khoảng cách giữa query và hard negative, đồng thời duy trì độ gần giữa query và positive.

Trong triển khai, vòng 2 vẫn sử dụng `MultipleNegativesRankingLoss`. Các hard negatives được lưu cùng dữ liệu để phân tích và tái lập thực nghiệm. Mặc dù loss chính vẫn là MNRL, quá trình mining làm thay đổi phân phối mẫu huấn luyện theo hướng khó hơn, giúp mô hình học biểu diễn có tính phân biệt cao hơn.

### 3.6. Tích hợp vào chatbot RAG

Sau khi fine-tune, retriever được dùng để lập chỉ mục vector cho các đoạn giáo trình. Khi người dùng đặt câu hỏi, hệ thống thực hiện dense retrieval để lấy top-k đoạn liên quan. Các đoạn này có thể được kết hợp với BM25 hoặc reranking để tăng độ chính xác trước khi đưa vào mô hình sinh.

Mô hình sinh được yêu cầu trả lời dựa trên ngữ cảnh được truy xuất. Nếu ngữ cảnh không chứa đủ thông tin, hệ thống cần phản hồi rằng chưa tìm thấy căn cứ phù hợp. Thiết kế này giúp giảm nguy cơ hallucination và tạo điều kiện đánh giá bằng RAGAS.

## Cấu hình fine-tuning4. Dữ liệu và độ đo đánh giá

### 4.1. Dữ liệu sử dụng

Dữ liệu tri thức của ứng dụng UniPolis gồm 6 giáo trình lý luận chính trị tiếng Việt, bao phủ các môn Lịch sử Đảng Cộng sản Việt Nam, Pháp luật đại cương, Triết học Mác-Lênin, Kinh tế Chính trị Mác-Lênin, Chủ nghĩa Xã hội Khoa học và Tư tưởng Hồ Chí Minh.

Đối với thực nghiệm fine-tuning retriever, dữ liệu được xây dựng từ ba môn có tập câu hỏi và đoạn liên quan đầy đủ: Lịch sử Đảng Cộng sản Việt Nam, Pháp luật đại cương và Triết học Mác-Lênin. Tổng số cặp query-passage là 4014. Bảng 1 trình bày số lượng mẫu trong từng split.


| Split      | Số mẫu |
| ---------- | ------ |
| Train      | 2787   |
| Validation | 616    |
| Test       | 611    |


Trong test set, câu hỏi được phân thành ba nhóm ý định. Bảng 2 trình bày phân bố của các nhóm này.


| Loại câu hỏi | Số mẫu |
| ------------ | ------ |
| Factual      | 191    |
| Relational   | 212    |
| Applicative  | 208    |


### 4.2. Độ đo đánh giá retrieval

Đánh giá retrieval sử dụng hai nhóm độ đo chính là MRR@10 và Recall@k.

MRR@10 đo thứ hạng của đoạn đúng đầu tiên trong top 10 kết quả. Nếu đoạn đúng xuất hiện ở vị trí đầu tiên, reciprocal rank bằng 1. Nếu xuất hiện ở vị trí thứ 2, giá trị bằng 1/2. Nếu đoạn đúng không xuất hiện trong top 10, giá trị bằng 0. MRR@10 là trung bình trên toàn bộ tập câu hỏi.

Recall@k đo tỷ lệ câu hỏi mà đoạn đúng xuất hiện trong top-k kết quả. Trong đồ án, các giá trị k gồm 1, 5 và 10. Recall@1 phản ánh khả năng đưa đoạn đúng lên vị trí đầu tiên, còn Recall@5 và Recall@10 phản ánh khả năng bao phủ đoạn đúng trong tập ngữ cảnh ứng viên.

### 4.3. Độ đo đánh giá RAG downstream

Đối với chatbot RAG, đồ án sử dụng RAGAS để đánh giá chất lượng đầu cuối. Các tiêu chí chính được trình bày ở Bảng 6.


| Tiêu chí          | Ý nghĩa                                                           |
| ----------------- | ----------------------------------------------------------------- |
| Faithfulness      | Mức độ câu trả lời được hỗ trợ bởi ngữ cảnh truy xuất             |
| Answer Relevancy  | Mức độ câu trả lời phù hợp với câu hỏi                            |
| Context Precision | Tỷ lệ ngữ cảnh truy xuất thật sự hữu ích trong các đoạn được chọn |
| Context Recall    | Mức độ ngữ cảnh truy xuất bao phủ thông tin cần thiết để trả lời  |


Các tiêu chí này bổ sung cho đánh giá retrieval. MRR và Recall cho biết retriever có tìm được đoạn đúng hay không, trong khi RAGAS đánh giá tác động của ngữ cảnh truy xuất đến câu trả lời cuối cùng.

## 5. Thực nghiệm truy xuất

### 5.1. Thiết lập thực nghiệm

Thực nghiệm chính so sánh ba phiên bản của Vietnamese Bi-Encoder. Phiên bản thứ nhất là mô hình gốc chưa fine-tune. Phiên bản thứ hai là mô hình sau khi fine-tune bằng `MultipleNegativesRankingLoss`. Phiên bản thứ ba là mô hình tiếp tục được huấn luyện sau Hard Negative Mining.

Tập validation được dùng để theo dõi quá trình huấn luyện và chọn checkpoint. Tập test chỉ được dùng để báo cáo kết quả cuối. Cách thiết lập này giúp hạn chế việc điều chỉnh mô hình dựa trên test set.

### 5.2. Cấu hình fine-tuning

Mô hình nền được sử dụng trong thực nghiệm chính là Vietnamese Bi-Encoder. Đây là mô hình Sentence Transformer được huấn luyện trước cho tiếng Việt và có thể mã hóa câu hỏi cũng như đoạn văn thành embedding 768 chiều. Mô hình được fine-tune theo hai giai đoạn. Giai đoạn 1 tối ưu trực tiếp các cặp query-positive bằng `MultipleNegativesRankingLoss`. Giai đoạn 2 sử dụng checkpoint tốt nhất của giai đoạn 1 để khai thác hard negatives, sau đó tiếp tục fine-tune với tập dữ liệu khó hơn.

Bảng 5 trình bày cấu hình huấn luyện chính.


| Thành phần                 | Cấu hình                                                  |
| -------------------------- | --------------------------------------------------------- |
| Mô hình nền                | Vietnamese Bi-Encoder                                     |
| Kiến trúc                  | Bi-Encoder / Sentence Transformer                         |
| Chiều embedding            | 768                                                       |
| Max sequence length        | 256 tokens                                                |
| Dữ liệu train/val/test     | 2787 / 616 / 611 cặp query-passage                        |
| Đơn vị chia dữ liệu        | `(source, chapter)`                                       |
| Loss giai đoạn 1           | `MultipleNegativesRankingLoss`                            |
| Epochs giai đoạn 1         | 3                                                         |
| Learning rate giai đoạn 1  | `1e-5`                                                    |
| Loss giai đoạn 2           | `MultipleNegativesRankingLoss` với hard negatives đã mine |
| Epochs giai đoạn 2         | 2                                                         |
| Learning rate giai đoạn 2  | `5e-6`                                                    |
| Batch size                 | 32                                                        |
| Warmup ratio               | 0.10                                                      |
| Weight decay               | 0.01                                                      |
| Evaluation steps           | 100                                                       |
| Save steps                 | 100                                                       |
| Metric chọn checkpoint     | Validation MRR@10                                         |
| Hard negative mining top-k | 30                                                        |
| Mining batch size          | 64                                                        |


Learning rate ở giai đoạn 2 được đặt nhỏ hơn giai đoạn 1 vì mô hình đã học được biểu diễn ban đầu và chỉ cần tinh chỉnh thêm trên các negative khó. Cách này giúp giảm nguy cơ làm lệch mạnh không gian embedding đã học ở giai đoạn đầu. Warmup được dùng trong 10% tổng số bước huấn luyện để tránh cập nhật quá lớn ở giai đoạn đầu, khi optimizer chưa ổn định.

Trong giai đoạn 1, mỗi batch gồm các cặp `(query, positive passage)`. Với batch size 32, mỗi query có 31 in-batch negatives. Điều này giúp mô hình học xếp hạng mà không cần tạo negative thủ công cho từng mẫu. Sau giai đoạn 1, mô hình được dùng để truy xuất top-30 đoạn trong corpus train cho từng query. Đoạn sai có điểm similarity cao nhất được chọn làm hard negative. Giai đoạn 2 tiếp tục huấn luyện mô hình trên dữ liệu đã bổ sung hard negatives nhằm tăng khả năng phân biệt các đoạn gần nghĩa.

Việc chọn checkpoint dựa trên validation MRR@10 thay vì training loss. Lý do là training loss chỉ phản ánh độ phù hợp với batch huấn luyện, còn MRR@10 phản ánh trực tiếp mục tiêu xếp hạng của bài toán retrieval. Checkpoint cuối cùng được đánh giá một lần trên test set độc lập để báo cáo kết quả cuối.

### 5.3. Kết quả fine-tuning theo hai giai đoạn

Bảng 3 trình bày kết quả MRR@10 của ba mô hình.


| Mô hình                           | Val MRR@10 | Test MRR@10 | Cải thiện test so với base |
| --------------------------------- | ---------- | ----------- | -------------------------- |
| Base Vietnamese Bi-Encoder        | 0.4234     | 0.4178      | +0.0000                    |
| V1 - MultipleNegativesRankingLoss | 0.4658     | 0.4795      | +0.0617                    |
| V2 - Hard Negative Mining         | 0.4864     | 0.4982      | +0.0805                    |


Kết quả cho thấy `MultipleNegativesRankingLoss` giúp cải thiện đáng kể mô hình gốc. Test MRR@10 tăng từ 0.4178 lên 0.4795. Sau khi áp dụng Hard Negative Mining, MRR@10 tiếp tục tăng lên 0.4982. Điều này chứng minh rằng hard negatives cung cấp tín hiệu huấn luyện bổ sung có ích, đặc biệt trong các trường hợp đoạn sai gần nghĩa với đoạn đúng.

### 5.4. Benchmark trên cùng test split 611 truy vấn

Ngoài đánh giá theo evaluator trong quá trình huấn luyện, đồ án thực hiện benchmark các mô hình embedding trên cùng test split 611 truy vấn. Kết quả được trình bày ở Bảng 4.


| Mô hình                  | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| ------------------------ | -------- | -------- | --------- | ------ |
| Vietnamese Bi-Encoder    | 0.6530   | 0.9051   | 0.9493    | 0.7603 |
| BGE-M3                   | 0.8052   | 0.9673   | 0.9902    | 0.8730 |
| Multilingual E5-Large    | 0.7971   | 0.9574   | 0.9869    | 0.8655 |
| Vietnamese Bi-Encoder V2 | 0.7676   | 0.9525   | 0.9738    | 0.8443 |


Trong benchmark này, Vietnamese Bi-Encoder V2 cải thiện rõ rệt so với Vietnamese Bi-Encoder gốc. Recall@1 tăng từ 0.6530 lên 0.7676, còn MRR@10 tăng từ 0.7603 lên 0.8443. Mức tăng tuyệt đối của MRR@10 là 0.0840, tương ứng cải thiện tương đối khoảng 11.05%.

Tuy nhiên, BGE-M3 và multilingual-E5-Large vẫn đạt kết quả cao hơn. Điều này cho thấy các mô hình embedding lớn, được huấn luyện trước trên tập dữ liệu quy mô lớn, vẫn có lợi thế mạnh. Kết quả này không làm giảm ý nghĩa của fine-tuning, mà cho thấy fine-tuning giúp mô hình gốc cùng họ thu hẹp khoảng cách đáng kể với các mô hình mạnh hơn.

### 5.5. Phân tích tác động của Hard Negative Mining

Hard Negative Mining cải thiện mô hình vì nó tập trung vào các trường hợp khó. Trong giáo trình, nhiều đoạn cùng chương có thể chứa các thuật ngữ tương tự, ví dụ cùng nói về một giai đoạn lịch sử, một phạm trù triết học hoặc một khái niệm pháp luật. Nếu chỉ dùng negative ngẫu nhiên hoặc in-batch negatives, mô hình có thể không gặp đủ các trường hợp gây nhầm lẫn.

Khi hard negatives được đưa vào huấn luyện, mô hình phải học cách phân biệt giữa đoạn “cùng chủ đề” và đoạn “trả lời đúng câu hỏi”. Đây là khác biệt quan trọng trong retrieval cho hỏi đáp. Một đoạn có thể liên quan về mặt chủ đề nhưng không chứa thông tin cần thiết để trả lời. Kết quả MRR@10 tăng sau vòng 2 cho thấy quá trình mining đã cải thiện khả năng xếp hạng các đoạn đúng lên vị trí cao hơn.

## 6. Ứng dụng RAG chatbot và đánh giá downstream

### 6.1. Vai trò của RAG trong đồ án

Trong đồ án này, RAG chatbot không phải là đóng góp kỹ thuật chính mà là ứng dụng minh họa cho retriever đã fine-tune. Lý do lựa chọn RAG là vì retrieval đóng vai trò trung tâm trong kiến trúc này. Nếu retriever tốt hơn, ngữ cảnh cung cấp cho mô hình sinh sẽ chính xác hơn, từ đó có khả năng cải thiện câu trả lời cuối cùng.

Ứng dụng UniPolis cho phép người học đặt câu hỏi về nội dung giáo trình lý luận chính trị. Hệ thống truy xuất các đoạn liên quan từ kho tài liệu, sau đó mô hình ngôn ngữ sinh câu trả lời dựa trên các đoạn này. Thiết kế prompt yêu cầu mô hình bám sát ngữ cảnh và hạn chế bổ sung thông tin ngoài tài liệu.

### 6.2. Pipeline RAG

Pipeline RAG gồm các bước: trích xuất văn bản từ giáo trình, làm sạch văn bản, chia đoạn có overlap, mã hóa đoạn văn bằng embedding model, lưu trữ vector, truy xuất top-k đoạn liên quan, tùy chọn reranking, và sinh câu trả lời từ ngữ cảnh.

Trong các thử nghiệm downstream, có thể so sánh các cấu hình retriever khác nhau trong cùng một pipeline RAG. Ví dụ, một cấu hình dùng Vietnamese Bi-Encoder gốc, một cấu hình dùng Vietnamese Bi-Encoder V2 sau Hard Negative Mining, và một cấu hình dùng BGE-M3. Cách so sánh này giúp xác định liệu cải thiện retrieval có chuyển hóa thành cải thiện chất lượng trả lời hay không.

### 6.3. Đánh giá bằng RAGAS

RAGAS được sử dụng để đánh giá câu trả lời của chatbot ở cấp độ đầu cuối. Bộ câu hỏi đánh giá cần bao gồm câu hỏi factual, relational và applicative để phản ánh nhiều loại nhu cầu học tập. Với mỗi câu hỏi, hệ thống lưu lại câu trả lời sinh ra và các đoạn ngữ cảnh được truy xuất.

Faithfulness đánh giá liệu câu trả lời có được hỗ trợ bởi ngữ cảnh hay không. Đây là tiêu chí quan trọng vì hệ thống hỏi đáp giáo trình cần tránh sinh thông tin không có trong tài liệu. Answer Relevancy đánh giá mức độ câu trả lời đi đúng trọng tâm câu hỏi. Context Precision cho biết các đoạn truy xuất có hữu ích hay không, còn Context Recall đo mức độ các đoạn truy xuất bao phủ thông tin cần thiết.

Kết quả RAGAS sẽ được dùng như đánh giá downstream cho retriever. Nếu retriever fine-tuned có Context Precision hoặc Context Recall cao hơn, điều đó cho thấy mô hình truy xuất đã cải thiện chất lượng ngữ cảnh. Nếu Faithfulness và Answer Relevancy cũng tăng, có thể kết luận rằng cải thiện retrieval có tác động tích cực đến chất lượng trả lời cuối cùng.

### 6.4. Kỳ vọng thực nghiệm downstream

Từ kết quả retrieval, có thể kỳ vọng mô hình Vietnamese Bi-Encoder V2 sẽ cải thiện chất lượng RAG so với Vietnamese Bi-Encoder gốc. Vì V2 có Recall@1 và MRR@10 cao hơn, các đoạn đúng có nhiều khả năng xuất hiện ở vị trí đầu trong context. Điều này đặc biệt quan trọng khi mô hình sinh chỉ nhận một số đoạn giới hạn.

Tuy nhiên, không thể mặc định rằng retrieval tốt hơn luôn làm RAGAS tốt hơn trên mọi tiêu chí. Chất lượng câu trả lời còn phụ thuộc vào prompt, mô hình sinh, độ dài context và cách kết hợp các đoạn truy xuất. Vì vậy, đánh giá RAGAS là cần thiết để kiểm tra tác động thực tế thay vì chỉ suy luận từ metrics retrieval.

## 7. Kết luận

Đồ án đã chuyển trọng tâm từ việc xây dựng một hệ thống RAG tổng quát sang nghiên cứu và cải thiện thành phần retriever bằng kỹ thuật học sâu. Cụ thể, đồ án fine-tune Vietnamese Bi-Encoder cho truy xuất ngữ nghĩa tiếng Việt bằng `MultipleNegativesRankingLoss` và Hard Negative Mining.

Kết quả thực nghiệm cho thấy fine-tuning bằng `MultipleNegativesRankingLoss` giúp cải thiện rõ rệt MRR@10 so với mô hình gốc. Khi bổ sung Hard Negative Mining, mô hình tiếp tục cải thiện, cho thấy negative khó có vai trò quan trọng trong việc học biểu diễn ngữ nghĩa cho retrieval. Trong benchmark trên cùng test split, Vietnamese Bi-Encoder V2 đạt MRR@10 0.8443, tăng 11.05% so với mô hình gốc cùng họ.

So với các embedding model mạnh như BGE-M3 và multilingual-E5-Large, mô hình fine-tuned chưa đạt kết quả cao nhất. Tuy nhiên, kết quả vẫn có ý nghĩa vì nó chứng minh rằng fine-tuning theo miền dữ liệu có thể cải thiện đáng kể một retriever tiếng Việt và thu hẹp khoảng cách với các mô hình lớn hơn.

UniPolis được xây dựng như một ứng dụng downstream của retriever đã fine-tune. Thông qua RAG chatbot và đánh giá RAGAS, đồ án có thể kiểm tra liệu cải thiện retrieval có dẫn đến cải thiện chất lượng hỏi đáp đầu cuối hay không. Đây là hướng đánh giá phù hợp vì nó kết nối đóng góp học sâu với một bài toán ứng dụng thực tế.

Trong tương lai, đồ án có thể mở rộng dữ liệu huấn luyện cho đủ 6 giáo trình, thử fine-tuning các mô hình embedding mạnh hơn như BGE-M3 nếu tài nguyên cho phép, bổ sung nhiều hard negatives cho mỗi query, và hoàn thiện bộ đánh giá RAGAS với ground-truth answer và ground-truth context được kiểm duyệt thủ công.

## 8. Bảng phân công


| Nhiệm vụ                                                     | Thành viên 1 | Thành viên 2 | Thành viên 3 | Thành viên 4 |
| ------------------------------------------------------------ | ------------ | ------------ | ------------ | ------------ |
| Tìm hiểu Dense Retrieval, Bi-Encoder và contrastive learning | X            | X            | X            | X            |
| Xây dựng dữ liệu query-passage và chia train/val/test        | X            | X            |              |              |
| Fine-tuning bằng MultipleNegativesRankingLoss                |              | X            | X            |              |
| Khai thác và huấn luyện với Hard Negative Mining             |              | X            | X            |              |
| Benchmark retrieval và phân tích kết quả                     | X            | X            | X            |              |
| Xây dựng ứng dụng RAG chatbot UniPolis                       | X            |              | X            |              |
| Đánh giá downstream bằng RAGAS                               | X            | X            |              | X            |
| Viết báo cáo và chuẩn bị thuyết trình                        | X            | X            | X            | X            |


Nhóm có thể thay các cột thành họ tên thành viên thực tế khi hoàn thiện bản nộp cuối cùng.

## 9. Tài liệu tham khảo

[1] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP, 2019.

[2] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020.

[3] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.

[4] BAAI. BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings, 2024.

[5] Nandan Thakur, Nils Reimers, Andreas Rucklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks, 2021.

[6] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL, 2024.

[7] Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR, 2020.

[8] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise Zero-Shot Dense Retrieval without Relevance Labels. ACL, 2023.