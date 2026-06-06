# Model weights — tải riêng

Trọng lượng model **không** nằm trong repo Git vì dung lượng ~517 MB.

## Vietnamese Bi-Encoder V2 (hard negative mining)

| Thuộc tính | Giá trị |
|------------|---------|
| Base model | `bkai-foundation-models/vietnamese-bi-encoder` |
| Artifact | `vietnamese-bi-encoder-v2-hnm` |
| Kaggle dataset | [`dangvy1507/models`](https://www.kaggle.com/datasets/dangvy1507/models) |

### Cách đặt model sau khi tải

Chọn **một** trong các vị trí sau (repo tự nhận diện):

```text
models/bi_encoder_hnm_v2/vietnamese-bi-encoder-v2-hnm/
```

hoặc

```text
bi-encoder-finetuned/models/bi_encoder_hnm_v2/vietnamese-bi-encoder-v2-hnm/
```

Hoặc set biến môi trường trong `.env`:

```bash
UNIPOLIS_VI_BI_ENCODER_PATH=/absolute/path/to/vietnamese-bi-encoder-v2-hnm
```

### Metadata & kết quả huấn luyện (đã có trong repo)

- `bi_encoder_hnm_v2/training_metadata.json` — cấu hình split, hyperparameter, MRR@10
- `bi_encoder_hnm_v2/artifacts/bi_encoder_hnm_eval_summary.json` — so sánh base / V1 / V2

### Dùng trong demo

Sau khi có model + Chroma build bằng cùng embedding, chọn pipeline **Quality · VN Bi-Encoder (FT)** trên Streamlit.
