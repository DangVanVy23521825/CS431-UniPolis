# Hướng Dẫn Fine-Tuning Bi-Encoder Cho RAG - Lý Luận Chính Trị

## 📋 Tóm Tắt Thực Thi

Cho dự án RAG trên các tài liệu Lý Luận Chính Trị bậc đại học:

- **Model cơ sở**: `bkai-foundation-models/vietnamese-bi-encoder`
- **Phương pháp fine-tuning**: **Contrastive Learning** (Triplet Loss hoặc In-Batch Negatives)
- **Thời lượng huấn luyện**: 3-5 epochs trên 50K-100K cặp dữ liệu
- **Hardware tối thiểu**: GPU 12GB VRAM (RTX 3060 trở lên)

---

## 1. Phân Tích Dữ Liệu Input

### Tài Liệu Phân Tích
- **File**: `lich-su-dang-cong-san-viet-nam-giao-trinh.pdf`
- **Loại**: Giáo trình đại học, nội dung lý luận chính trị
- **Tổng trang**: 223 trang
- **Đặc điểm**:
  - Nội dung có cấu trúc rõ ràng (chương, mục, tiểu mục)
  - Dàn trải khái niệm, sự kiện, luận điểm lý luận
  - Yêu cầu phân biệt mối liên hệ giữa các khái niệm trừu tượng
  - Đôi khi có nội dung lịch sử, chính trị, ý thức hình cần liên hệ tổng hợp

### Thách Thức Đặc Hữu Của Domain Này

1. **Sự trừu tượng cao**: Khái niệm như "Cương lĩnh", "đường lối", "tư tưởng" không có tương tự ngữ nghĩa trong các tập dữ liệu general
2. **Thuật ngữ tuyên truyền + khoa học**: Kết hợp giữa ngôn ngữ chính thức/khoa học và nôi dung ý thức hình
3. **Yêu cầu quan hệ ngữ cảnh sâu**: Một câu hỏi có thể cần hiểu nhiều đoạn văn liên tiếp để trả lời

---

## 2. Phương Pháp Fine-Tuning Được Khuyến Nghị

### 2.1 Contrastive Learning vs Các Lựa Chọn Khác

| Phương pháp | Ưu điểm | Hạn chế | Phù hợp? |
|-------------|---------|--------|---------|
| **Contrastive Learning** (Triplet/In-Batch) | Tối ưu hóa trực tiếp khoảng cách embedding, hiệu quả từ 5K-100K mẫu | Cần cặp dữ liệu (query-document) có chất lượng tốt | ✅ **ĐỦ TỐT** |
| **Supervised Fine-tuning** | Đơn giản, không cần tạo cặp | Cần nhãn 3+ lớp (too abstract cho embeddings) | ❌ Không phù hợp |
| **Hard Negatives Mining** | Cải thiện khả năng phân biệt | Phức tạp, cần 2 vòng huấn luyện | ⚠️ Nâng cao (vòng thứ 2) |
| **In-Batch Negatives** | Đơn giản, hiệu quả với batch lớn | Cần batch ≥ 256-512 | ✅ **KHUYẾN NGHỊ** |
| **Sentence-Transformer** (SBERT) | Nhúng sẵn pipeline training | Cần matching ground truth | ✅ **KHUYẾN NGHỊ** |

**⭐ KHUYẾN NGHỊ CHÍNH**: Sử dụng **`sentence-transformers` library** với loss function **`MultipleNegativesRankingLoss`** (tương tự In-Batch Negatives)

---

## 3. Quy Trình Sinh Dữ Liệu Fine-Tuning

### 3.1 Chiến Lược Toàn Thể

```
PDF → Chunk & Segment → Query-Document Pairs → Validation → Training Data
                              ↓
                    (Sinh tự động + Manual)
```

### 3.2 Bước 1: Chunking & Tiền Xử Lý

#### 3.2.1 Cấp độ Chunk

Vì domain này yêu cầu ngữ cảnh sâu, khuyến nghị **3 cấp độ**:

```
Cấp 1: Tiểu mục (Sub-Section) ~200-400 từ
       → Giữ ngữ cảnh chương
       
Cấp 2: Đoạn văn + 1-2 đoạn liền kề ~100-200 từ
       → Đơn vị chuẩn cho retrieval
       
Cấp 3: Câu/nhóm câu ~30-80 từ
       → Cho fine-grained matching
```

#### 3.2.2 Cách Thực Hiện (Python)

```python
import re
from typing import List, Tuple

class PoliticalTextChunker:
    def __init__(self, min_chunk_tokens=50, max_chunk_tokens=200):
        self.min_tokens = min_chunk_tokens
        self.max_tokens = max_chunk_tokens
    
    def chunk_by_section(self, pdf_text: str) -> List[dict]:
        """
        Tách văn bản PDF theo cấu trúc Chương/Mục/Tiểu mục
        """
        # Pattern cho tiêu đề (cần điều chỉnh theo từng tài liệu)
        chapter_pattern = r'(?:Chương|Phần)\s+[\IVX0-9]+[:.]*\s*(.+?)(?=(?:Chương|Phần)|\Z)'
        section_pattern = r'(?:I+|[0-9]+)[:.)\s]+(.+?)(?=(?:I+|[0-9]+)[:.)]|\Z)'
        
        chunks = []
        doc_id = 1
        
        # Tách chương
        chapters = re.finditer(chapter_pattern, pdf_text, re.IGNORECASE | re.DOTALL)
        for chapter_match in chapters:
            chapter_text = chapter_match.group(0)
            chapter_title = chapter_match.group(1)[:100]  # Giới hạn độ dài
            
            # Tách mục trong chương
            sections = re.finditer(section_pattern, chapter_text)
            for section_match in sections:
                section_text = section_match.group(0)
                
                # Tách đoạn văn
                paragraphs = [p.strip() 
                              for p in section_text.split('\n\n') 
                              if p.strip()]
                
                # Kết hợp đoạn văn thành chunks
                for i in range(0, len(paragraphs), 2):
                    chunk_content = ' '.join(paragraphs[i:i+2])
                    
                    if len(chunk_content.split()) >= self.min_tokens:
                        chunks.append({
                            'id': f'doc_{doc_id}',
                            'chapter': chapter_title,
                            'section': section_match.group(1)[:80] if section_match else '',
                            'content': chunk_content,
                            'word_count': len(chunk_content.split())
                        })
                        doc_id += 1
        
        return chunks
```

---

### 3.3 Bước 2: Sinh Query-Document Pairs (Cặp Dữ Liệu)

#### 3.3.1 Phương Pháp 1: Sinh Tự Động (75% dữ liệu)

**Sử dụng LLM (Claude hoặc Open-source) để sinh câu hỏi**

```python
from anthropic import Anthropic

def generate_queries_for_chunk(chunk_content: str, num_queries=3) -> List[str]:
    """
    Sinh 3-5 câu hỏi đa dạng từ 1 chunk
    """
    client = Anthropic()
    
    prompt = f"""Dựa trên nội dung sau về Lý luận Chính trị, hãy sinh {num_queries} câu hỏi:
- 1 câu hỏi trực tiếp (định nghĩa, khái niệm)
- 1 câu hỏi liên hệ (mối quan hệ, so sánh)
- 1 câu hỏi ứng dụng (ví dụ, hệ quả)

Nội dung:
{chunk_content[:500]}

Format: Trả về CHỈ 1 câu hỏi trên 1 dòng, không có số thứ tự."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    queries = [q.strip() for q in message.content[0].text.strip().split('\n') 
               if q.strip()]
    return queries[:num_queries]


def create_training_pairs_auto(chunks: List[dict]) -> List[dict]:
    """
    Tạo training pairs từ chunks
    Output: [{"query": "...", "positive": "...", "negatives": ["...", "..."]}, ...]
    """
    training_pairs = []
    
    for i, chunk in enumerate(chunks):
        # Sinh queries cho chunk này (document dương)
        try:
            queries = generate_queries_for_chunk(chunk['content'], num_queries=2)
        except Exception as e:
            print(f"Error generating queries for chunk {i}: {e}")
            queries = []
        
        for query in queries:
            # Negative sampling: 2-3 chunks khác ngẫu nhiên
            import random
            negative_indices = random.sample(
                [j for j in range(len(chunks)) if j != i],
                k=min(3, len(chunks) - 1)
            )
            
            pair = {
                'query': query,
                'positive': chunk['content'],
                'positive_id': chunk['id'],
                'negatives': [chunks[j]['content'] for j in negative_indices],
                'negative_ids': [chunks[j]['id'] for j in negative_indices]
            }
            training_pairs.append(pair)
    
    return training_pairs
```

#### 3.3.2 Phương Pháp 2: Sinh Manual/Hybrid (25% dữ liệu)

**Tạo gold standard queries từ câu hỏi thực tế**

```python
# Ví dụ câu hỏi tạo thủ công cho domain Lịch sử Đảng
SEED_QUERIES = [
    # Khái niệm cơ bản
    ("Đảng Cộng Sản Việt Nam được sáng lập vào năm nào?", 
     "ra đời, phát triển, hoạt động lãnh đạo, 3-2-1930"),
    
    ("Cương lĩnh chính trị đầu tiên của Đảng được đề ra khi nào?", 
     "Cương lĩnh chính trị đầu tiên, 2-1930"),
    
    ("Đảng Cộng Sản lấy gì làm nền tảng tư tưởng?", 
     "chủ nghĩa Mác-Lênin, tư tưởng Hồ Chí Minh, nền tảng tư tưởng"),
    
    # Mối quan hệ, so sánh
    ("Sự khác biệt giữa sự kiện lịch sử Đảng và sự kiện lịch sử dân tộc là gì?", 
     "phân biệt sự kiện lịch sử Đảng, lịch sử dân tộc, lịch sử quân sự"),
    
    ("Tại sao đường lối đúng đắn là điều kiện quyết định thắng lợi?", 
     "đường lối đúng đắn, điều kiện, thắng lợi, cách mạng"),
    
    # Ứng dụng, hệ quả
    ("Ban Nghiên cứu Lịch sử Đảng Trung ương được thành lập với mục đích gì?",
     "Ban Nghiên cứu Lịch sử Đảng, Viện Lịch sử Đảng, 1962"),
]

def expand_manual_queries(seed_queries: List[Tuple[str, str]],
                          chunks: List[dict]) -> List[dict]:
    """
    Match manual queries với chunks dựa trên keywords
    """
    import difflib
    
    manual_pairs = []
    
    for query, keywords in seed_queries:
        # Tìm chunk phù hợp nhất
        best_match = None
        best_score = 0
        
        for chunk in chunks:
            # Scoring dựa trên keyword overlap
            chunk_lower = chunk['content'].lower()
            keyword_list = keywords.split(',')
            
            score = sum(keyword.lower() in chunk_lower 
                       for keyword in keyword_list) / len(keyword_list)
            
            if score > best_score:
                best_score = score
                best_match = chunk
        
        if best_match and best_score > 0.3:  # Threshold
            # Negative sampling
            negative_indices = random.sample(
                [i for i, c in enumerate(chunks) if c['id'] != best_match['id']],
                k=min(3, len(chunks) - 1)
            )
            
            manual_pairs.append({
                'query': query,
                'positive': best_match['content'],
                'positive_id': best_match['id'],
                'negatives': [chunks[i]['content'] for i in negative_indices],
                'negative_ids': [chunks[i]['id'] for i in negative_indices],
                'source': 'manual_gold'
            })
    
    return manual_pairs
```

---

### 3.4 Bước 3: Format Dữ Liệu Cho Training

```python
import json

def format_for_sentence_transformers(training_pairs: List[dict]) -> str:
    """
    Format cho thư viện sentence-transformers
    Format: (query, positive, negative1, negative2, ...)
    """
    formatted = []
    
    for pair in training_pairs:
        row = [
            pair['query'],
            pair['positive']
        ] + pair['negatives']
        
        formatted.append(row)
    
    # Lưu thành JSON Lines
    with open('training_data.jsonl', 'w', encoding='utf-8') as f:
        for row in formatted:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    return len(formatted)
```

---

### 3.5 Tóm Tắt Qui Trình Sinh Dữ Liệu

```
📊 KÍCH THƯỚC DỰ KIẾN:

Input:  223 trang PDF
        ↓
Chunks: ~1,500-2,000 chunks (Cấp 2: 100-200 từ)
        ↓
Queries: 
- Auto: 1,500 × 2 queries = 3,000 queries
- Manual: 100-200 seed queries → ~150 match tốt
- Total: ~3,150 training examples
        ↓
After cleaning & dedup: ~2,500-3,000 pairs (80% quality)

🎯 CÁCH PHÂN BỔ:
- 70%: Training (1,750-2,100 pairs)
- 15%: Validation (375-450 pairs)
- 15%: Test (375-450 pairs)
```

---

## 4. Cấu Hình & Huấn Luyện Fine-Tuning

### 4.1 Setup Environment

```bash
pip install sentence-transformers torch transformers datasets accelerate

# Phiên bản khuyến nghị
# sentence-transformers>=2.7.0
# torch>=2.0.0
```

### 4.2 Script Huấn Luyện (PyTorch)

```python
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import json

def train_bi_encoder():
    # ========== 1. Load base model ==========
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    model = SentenceTransformer(model_name)
    
    # Cấu hình model
    print(f"Model pooling: {model.get_sentence_embedding_dimension()}")
    print(f"Input max length: {model.get_max_seq_length()}")
    
    # ========== 2. Load dữ liệu ==========
    train_examples = []
    
    with open('training_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            # row = [query, positive, negative1, negative2, ...]
            query = row[0]
            positive = row[1]
            negatives = row[2:]
            
            # Tạo InputExample
            train_examples.append(InputExample(
                texts=[query, positive] + negatives,
                label=0  # Positive là index 1 (0-indexed)
            ))
    
    print(f"Loaded {len(train_examples)} training examples")
    
    # ========== 3. Tạo DataLoader ==========
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=32  # Điều chỉnh theo GPU memory
    )
    
    # ========== 4. Chọn Loss Function ==========
    # MultipleNegativesRankingLoss = In-Batch Negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # ========== 5. Cấu hình training ==========
    warmup_steps = int(len(train_dataloader) * 0.1)  # 10% warmup
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=4,
        evaluation_steps=500,
        evaluator=None,  # Sẽ thêm sau
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 2e-5},  # Learning rate
        output_path='./models/vietnamese-bi-encoder-finetuned',
        save_best_model=True,
        checkpoint_path='./models/checkpoint',
        checkpoint_save_steps=1000,
        checkpoint_save_total_limit=3
    )
    
    print("✅ Training completed!")
    return model

if __name__ == '__main__':
    train_bi_encoder()
```

### 4.3 Hyperparameters Chi Tiết

| Parameter | Giá trị | Giải thích |
|-----------|--------|-----------|
| **Batch size** | 32-64 | Điều chỉnh theo GPU (32GB = 64, 12GB = 32) |
| **Learning rate** | 2e-5 | Tiêu chuẩn cho fine-tuning |
| **Epochs** | 3-5 | 5 epochs = ~50K examples × 5 = 250K updates |
| **Warmup steps** | 10% của total | Tăng dần learning rate |
| **Weight decay** | 0.01 | Regularization |
| **Max seq length** | 384 (Vietnamese-bi-encoder) | Giữ mặc định |
| **Validation steps** | 500 | Eval mỗi 500 steps |

### 4.4 Loss Function Chi Tiết

```python
# MultipleNegativesRankingLoss (khuyến nghị)
# - Giả định: positive của query A là negative cho query B nếu khác batch
# - Tối ưu hóa: max margin between positive và negatives
# - Công thức: loss = -log( exp(sim(q,p)) / Σ exp(sim(q,d)) )
#              trong đó d = p + tất cả negatives khác

# Thay thế: TripletLoss (nếu muốn hard negatives)
from sentence_transformers.losses import TripletLoss
train_loss = TripletLoss(model, triplet_margin=0.5)
```

---

## 5. Evaluation & Validation

### 5.1 Metrics Để Theo Dõi

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import numpy as np

def evaluate_model(model, val_pairs: List[dict]) -> dict:
    """
    Đánh giá model trên validation set
    Metrics: MRR, NDCG@10, Recall@1/10
    """
    
    # Tính embeddings
    queries = [p['query'] for p in val_pairs]
    documents = [p['positive'] for p in val_pairs]
    
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    
    # Cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(
        query_embeddings.unsqueeze(1),
        doc_embeddings.unsqueeze(0),
        dim=2
    )
    
    # Tính MRR, Recall@k
    mrr_scores = []
    recall_1 = 0
    recall_10 = 0
    
    for i in range(len(val_pairs)):
        # Rank documents for query i
        ranked = torch.argsort(similarity_matrix[i], descending=True)
        
        # Position của positive (assuming positive_id đúng)
        positive_id = val_pairs[i]['positive_id']
        rank_position = (ranked == i).nonzero(as_tuple=True)[0].item() + 1
        
        # MRR
        mrr_scores.append(1.0 / rank_position)
        
        # Recall
        if rank_position <= 1:
            recall_1 += 1
        if rank_position <= 10:
            recall_10 += 1
    
    metrics = {
        'MRR': np.mean(mrr_scores),
        'Recall@1': recall_1 / len(val_pairs),
        'Recall@10': recall_10 / len(val_pairs),
    }
    
    print(f"📊 Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics
```

### 5.2 A/B Testing: Before vs After

```python
def compare_models(model_base, model_finetuned, test_pairs):
    """Compare base model vs finetuned"""
    
    queries = [p['query'] for p in test_pairs]
    positives = [p['positive'] for p in test_pairs]
    
    # Scores
    base_scores = []
    ft_scores = []
    
    for query, positive in zip(queries, positives):
        # Base model
        q_emb_base = model_base.encode(query)
        p_emb_base = model_base.encode(positive)
        score_base = np.dot(q_emb_base, p_emb_base)
        base_scores.append(score_base)
        
        # Finetuned model
        q_emb_ft = model_finetuned.encode(query)
        p_emb_ft = model_finetuned.encode(positive)
        score_ft = np.dot(q_emb_ft, p_emb_ft)
        ft_scores.append(score_ft)
    
    improvement = (np.mean(ft_scores) - np.mean(base_scores)) / np.mean(base_scores) * 100
    
    print(f"✅ Improvement: +{improvement:.2f}%")
    print(f"   Base model avg score: {np.mean(base_scores):.4f}")
    print(f"   Finetuned model avg score: {np.mean(ft_scores):.4f}")
```

---

## 6. Triển Khai & Sử Dụng

### 6.1 Sử Dụng Model Finetuned Trong RAG

```python
from sentence_transformers import SentenceTransformer
import faiss

class PoliticalRAGRetriever:
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)
        self.index = None
        self.documents = []
    
    def build_index(self, documents: List[str]):
        """Xây dựng FAISS index từ documents"""
        embeddings = self.model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Tạo FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings (cho cosine similarity)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.documents = documents
        print(f"✅ Built index with {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents"""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = [
            (self.documents[idx], score)
            for idx, score in zip(indices[0], scores[0])
        ]
        return results

# Sử dụng
retriever = PoliticalRAGRetriever(
    './models/vietnamese-bi-encoder-finetuned'
)
retriever.build_index(all_documents)

# Query
results = retriever.retrieve(
    "Đảng Cộng Sản Việt Nam lấy gì làm nền tảng tư tưởng?",
    top_k=5
)
for doc, score in results:
    print(f"Score: {score:.4f}\n{doc[:200]}...\n")
```

---

## 7. Tối Ưu Hóa Advanced (Nâng Cao)

### 7.1 Hard Negatives Mining (Vòng 2)

```python
def mine_hard_negatives(model, train_pairs, top_k=10):
    """
    Tìm hard negatives: documents có high similarity nhưng không phải positive
    """
    hard_pairs = []
    
    for pair in train_pairs:
        query = pair['query']
        positive = pair['positive']
        all_negatives = pair['negatives']
        
        # Encode
        q_emb = model.encode(query)
        neg_embs = [model.encode(neg) for neg in all_negatives]
        
        # Tính similarity
        similarities = [np.dot(q_emb, neg_emb) for neg_emb in neg_embs]
        
        # Sort by similarity (hardest first)
        hard_indices = np.argsort(similarities)[::-1][:top_k]
        hard_negs = [all_negatives[i] for i in hard_indices]
        
        hard_pairs.append({
            'query': query,
            'positive': positive,
            'negatives': hard_negs  # Update với hard negatives
        })
    
    return hard_pairs

# Sử dụng
hard_pairs = mine_hard_negatives(model, train_pairs, top_k=5)
# Train lại với hard negatives (epochs 1-2)
```

### 7.2 Domain-Specific Augmentation

```python
def augment_queries(query: str) -> List[str]:
    """
    Sinh thêm câu hỏi tương tự từ 1 câu hỏi
    """
    augmented = [query]
    
    # Paraphrase 1: Rút gọn
    if len(query.split()) > 10:
        augmented.append(' '.join(query.split()[:8]) + '?')
    
    # Paraphrase 2: Thêm thuật ngữ miền
    domain_terms = {
        'Đảng': ['Đảng Cộng Sản Việt Nam', 'Tổ chức Đảng'],
        'thắng lợi': ['thành công', 'chiến thắng'],
        'cách mạng': ['cách mạng Việt Nam', 'phong trào cách mạng'],
    }
    
    for original, replacements in domain_terms.items():
        if original in query:
            for replacement in replacements:
                augmented.append(query.replace(original, replacement))
    
    return augmented[:5]  # Giới hạn
```

---

## 8. Troubleshooting & Best Practices

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-----------|----------|
| **Loss không giảm** | Learning rate quá cao | Giảm từ 2e-5 → 5e-6 |
| **Overfitting (train loss ↓, val loss ↑)** | Quá nhiều epochs | Dừng ở epoch 2-3 |
| **OOM (Out of Memory)** | Batch size quá lớn | Giảm từ 64 → 32 → 16 |
| **Eval score không cải thiện** | Dữ liệu training kém chất lượng | Kiểm tra negatives, review pairs |
| **Model quá slow** | Embedding dimension quá lớn | Sử dụng distillation (nâng cao) |

---

## 9. Timeline & Checklist

### Phase 1: Chuẩn Bị (1-2 tuần)
- [ ] Trích xuất & chunking toàn bộ 4-5 môn lý luận
- [ ] Sinh 50K-100K query-document pairs
- [ ] QA & cleaning dữ liệu
- [ ] Split train/val/test

### Phase 2: Fine-tuning (1 tuần)
- [ ] Setup environment & hardware
- [ ] Train base model (3-5 epochs)
- [ ] Đánh giá metrics
- [ ] Điều chỉnh hyperparameters

### Phase 3: Optimization (1 tuần)
- [ ] Hard negatives mining
- [ ] Domain-specific augmentation
- [ ] A/B testing
- [ ] Lựa chọn best checkpoint

### Phase 4: Deployment (1 tuần)
- [ ] Build FAISS index
- [ ] Integrate vào RAG pipeline
- [ ] Load test & monitoring
- [ ] Document & release

---

## 10. Tài Liệu Tham Khảo

### Code/Library
- **Sentence-Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Vietnamese Tokenizer**: `pyvi`, `underthesea`

### Papers
- Sentence-BERT: https://arxiv.org/abs/1908.10084
- In-Batch Negatives: https://arxiv.org/abs/2004.14666

### Vietnamese Models
- BKAI Foundation: https://huggingface.co/bkai-foundation-models
- PhoBERT: https://github.com/VinAI/PhoBERT

---

## 📝 Tóm Tắt Nhanh

```
🎯 Chọn phương pháp: Contrastive Learning (MultipleNegativesRankingLoss)

📊 Sinh dữ liệu:
   - Auto: Generate 3-5 queries per chunk (LLM)
   - Manual: 100-200 gold-standard queries
   - Total: 2,500-3,000 training pairs

⚙️  Huấn luyện:
   - Batch: 32-64
   - LR: 2e-5
   - Epochs: 3-5
   - Loss: MultipleNegativesRankingLoss

✅ Eval:
   - MRR, Recall@1, Recall@10
   - A/B test vs base model
   - Expect ~10-20% improvement

🚀 Deploy: FAISS index + SentenceTransformer
```

---

**Bản cập nhật: May 2026**
**Tác giả: Claude AI Assistant**
