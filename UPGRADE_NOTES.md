# Legal RAG Upgrade Notes

Bản nâng cấp này tập trung vào ba mục tiêu: tăng độ chính xác retrieval, giảm nhiễu pháp lý và giảm latency trên CPU.

## Thay đổi chính

1. Thêm `src/retrieval/legal_signals.py`
   - Nhận diện query quan hệ văn bản để route sang Graph.
   - Nhận diện query Điều/Khoản/số hiệu văn bản.
   - Nhận diện topic hôn nhân gia đình: quyền nuôi con, cấp dưỡng, tài sản, kết hôn, ly hôn, hộ tịch.
   - Tính điểm metadata-aware, current-law-aware, article-aware.

2. Sửa BM25
   - Không trả tài liệu có score bằng 0.
   - Thêm legal token expansion như `dieu_81`, `luat_hngd_2014`, `topic_custody`.

3. Nâng cấp Hybrid Retrieval
   - Dense + BM25 bằng RRF.
   - BM25 weight tăng để hợp legal keyword search.
   - Metadata boost ở fusion stage được kiểm soát bằng `fusion_metadata_weight`.

4. Nâng cấp Reranker
   - Force rerank cho auto/hybrid_rerank.
   - Boost mạnh đúng Điều/Khoản, số hiệu, Luật HNGĐ 2014.
   - Boost văn bản còn hiệu lực, phạt mạnh văn bản hết hiệu lực.
   - Boost topic đặc thù hôn nhân gia đình và giảm nhiễu topic không liên quan.

5. Sửa Auto Routing
   - Query quan hệ văn bản/sửa đổi/thay thế/bãi bỏ/hiệu lực -> Graph.
   - Query hỏi đáp pháp luật thông thường -> Hybrid_Rerank.
   - Auto không còn hành xử giống Graph cho mọi câu hỏi.

6. Tối ưu Graph
   - Mặc định `graph_max_hops=1`.
   - Giới hạn seed doc và expansion để giảm nhiễu/latency.
   - Chỉ dùng Graph là nhánh chính khi query thật sự cần quan hệ văn bản.

7. Nâng benchmark
   - In `route_counts` cho Auto để kiểm tra routing.
   - Hỗ trợ metric `ArticleHit@k` nếu test case có `relevant_articles`.
   - `generate_test_cases.py` bổ sung `relevant_articles` cho nhóm Điều luật.

## Cách chạy lại

```powershell
python scripts/ingest.py
python scripts/generate_test_cases.py
python scripts/run_benchmark.py
```

Nếu muốn kiểm tra nhanh Auto route:

```yaml
retrieval:
  log_retrieval: true
```

## Kỳ vọng

- `AUTO` nên tiệm cận hoặc vượt `HYBRID_RERANK`, không còn giống `GRAPH` toàn bộ.
- `exact-keyword-fact` và câu hỏi Điều/Khoản nên tăng rõ.
- `single-hop-semantic` không bị tụt như khi dùng Graph làm chính.
