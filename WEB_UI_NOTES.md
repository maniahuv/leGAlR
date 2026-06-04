# Simple Web UI Upgrade

Bản nâng cấp này thêm giao diện web đơn giản cho hệ thống Legal RAG.

## Cấu trúc giao diện

Giao diện được chia thành 3 khung:

1. **Bên trái**: lịch sử hội thoại, tạo cuộc trò chuyện mới, chọn strategy, top-k, bật/tắt sinh câu trả lời bằng LLM.
2. **Ở giữa**: khung chat hỏi đáp pháp luật.
3. **Bên phải**: panel tra cứu nhanh. Khi người dùng bôi đậm/chọn một từ hoặc cụm trong khung chat, hệ thống gọi `/api/keyword/lookup` để hiển thị các văn bản pháp luật liên quan, điều/khoản, trích đoạn và link nguồn nếu có.

## File mới

```txt
web/index.html
web/styles.css
web/app.js
WEB_UI_NOTES.md
```

## File đã sửa

```txt
src/api/main.py
src/api/routers/query.py
```

## API mới

### POST `/api/query`

Dùng cho hỏi đáp chat. Response có thêm:

```json
{
  "timings": {
    "retrieval_ms": 12.3,
    "context_ms": 0.2,
    "prompt_ms": 0.3,
    "llm_init_ms": 3.1,
    "llm_generation_ms": 2500.0,
    "total_ms": 2520.1
  },
  "error": null
}
```

Nếu Gemini lỗi quota/rate limit, API không crash mà trả về thông báo lỗi trong `error` và vẫn trả về `documents` đã truy xuất.

### POST `/api/keyword/lookup`

Dùng cho panel bên phải khi bôi đậm/chọn keyword.

Request:

```json
{
  "keyword": "Điều 55",
  "k": 5,
  "strategy": "dense"
}
```

Response trả về danh sách văn bản/chunk liên quan kèm metadata như `title`, `so_ky_hieu`, `article`, `source_url`, `snippet`.

## Cách chạy

```powershell
uvicorn src.api.main:app --reload
```

Mở:

```txt
http://127.0.0.1:8000/
```

Hoặc:

```txt
http://127.0.0.1:8000/web/index.html
```

## Gợi ý demo

- Dùng `strategy=dense`, `k=3` để có tốc độ tốt.
- Bật `generate` nếu còn quota Gemini.
- Tắt `generate` để demo retrieval nhanh và tránh lỗi quota.
- Bôi đậm/chọn các từ như `Điều 55`, `thuận tình ly hôn`, `quyền nuôi con`, `cấp dưỡng`, `tài sản chung` để xem panel phải tự tra cứu nguồn liên quan.
