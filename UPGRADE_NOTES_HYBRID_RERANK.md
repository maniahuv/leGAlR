# Hybrid / Hybrid Rerank Upgrade Notes

Bản này nâng cấp retrieval sau khi corpus Hôn nhân & Gia đình đã mở rộng lên nhiều nhóm văn bản: HN&GĐ lõi, Hộ tịch, Nuôi con nuôi, Bạo lực gia đình, xử phạt, tố tụng dân sự, dân sự.

## File đã sửa chính

- `src/retrieval/legal_signals.py`
  - Thêm `detect_legal_topic()` để nhận diện intent pháp lý.
  - Thêm intent-level boost cho các nhóm: thuận tình ly hôn, đơn phương ly hôn, quyền nuôi con, Điều 81/82/83/84, cấp dưỡng, tài sản vợ chồng, hộ tịch, nuôi con nuôi, bạo lực gia đình, xử phạt, tố tụng dân sự, dân sự, trẻ em.
  - Thêm `legal_intent_score()` và penalty chống nhiễu từ các văn bản lớn như BLTTDS, BLDS, NĐ 82/2020 khi query không hỏi đúng miền đó.

- `src/retrieval/hybrid.py`
  - Chỉnh Hybrid thành dense-preserving hybrid.
  - Giảm mặc định trọng số BM25, metadata; tăng vai trò Dense.
  - Bảo toàn một số kết quả Dense top đầu để BM25 không kéo nhiễu.

- `src/retrieval/reranker.py`
  - Thêm intent-aware rerank.
  - Thêm rank prior nhỏ để không phá kết quả Dense/Hybrid tốt.
  - Giảm tác động lexical overlap của các từ chung.

- `src/tools/retrieval_tools.py`
  - Sửa AUTO: graph cho câu hỏi quan hệ văn bản, dense cho câu hỏi thường.
  - Hybrid_rerank vẫn dùng được khi gọi explicit `strategy="hybrid_rerank"`.

- `configs/config.yaml`
  - Chỉnh weight/candidate pool/context/generation token.
  - `generation.max_tokens` giảm còn 512 để dễ hướng tới 3-5s.

- `src/evaluation/benchmark.py`, `scripts/run_benchmark.py`
  - Lưu thêm `latency_ms`, `hit`, `article_hit` theo từng case.
  - Thêm `--show-failures` để in case fail.

- `test.py`
  - Đo nhiều lần trong cùng process sau warm-up.
  - Mặc định tắt LLM để đo retrieval/prompt, tránh quota Gemini.

## Lệnh gợi ý

```powershell
python scripts/run_benchmark.py --show-failures
python test.py
$env:TEST_STRATEGY="hybrid_rerank"; python test.py
$env:TEST_ENABLE_LLM="1"; python test.py
```
