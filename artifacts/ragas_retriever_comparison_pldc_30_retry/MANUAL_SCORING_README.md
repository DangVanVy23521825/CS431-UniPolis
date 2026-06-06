# Chấm Answer Relevancy thủ công

## File
- `manual_answer_relevancy_all.csv` — gộp base + V2
- `manual_answer_relevancy_<model>.csv` — từng model

## Cách chấm (cột `manual_relevancy`)
Chỉ cần đọc `question` + `answer` (cột ground_truth/contexts chỉ tham khảo).

| Điểm | Ý nghĩa |
|------|---------|
| 1.0 | Trả lời đúng trọng tâm, không lan man |
| 0.5 | Có liên quan nhưng thiếu/sai trọng tâm hoặc thừa ý |
| 0.0 | Lạc đề hoặc không trả lời câu hỏi |

## Tính trung bình
Sau khi điền xong, chạy:

```bash
python scripts/summarize_manual_answer_relevancy.py --csv artifacts/.../manual_answer_relevancy_all.csv
```
