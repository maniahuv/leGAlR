# Vietnamese Legal RAG - Upgraded

Repo này đã được nâng cấp theo hướng **Legal Modular RAG** cho dữ liệu pháp luật Việt Nam, đặc biệt là miền **hôn nhân và gia đình**.

## Nâng cấp chính

- Ingest đúng cấu trúc dataset `th1nhng0/vietnamese-legal-documents`: `metadata + content + relationships`.
- Lọc miền hôn nhân gia đình bằng cả metadata và nội dung, không chỉ dựa vào title.
- Cleaner giữ lại cấu trúc pháp luật `Chương/Mục/Điều/Khoản/Điểm`.
- Chunker ưu tiên tách theo `Điều`, tránh phá cấu trúc văn bản luật.
- BM25 tiếng Việt dùng `underthesea` tokenizer.
- Hybrid Search dùng **Reciprocal Rank Fusion (RRF)** thay vì merge thủ công.
- Reranker rule-based pháp lý có điều kiện, hỗ trợ Điều/Khoản/Số hiệu/hiệu lực.
- Graph-guided retrieval sửa lỗi graph docs bị cắt khỏi top-k; có metadata `graph_path`.
- API FastAPI chạy được với endpoint `/api/query`.
- Benchmark có `dense`, `hybrid`, `hybrid_rerank`, `graph`, `auto`.
- Sinh test case chỉ từ các `doc_id` đã index nếu có manifest, giảm lỗi gold không nằm trong index.

## Cài đặt

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Điền `GOOGLE_API_KEY` vào `.env` nếu muốn sinh câu trả lời bằng Gemini. Retrieval/benchmark có thể chạy không cần LLM key.

## Chạy ingest

```bash
python scripts/ingest.py
```

Pipeline:

```txt
metadata + content + relationships
→ lọc domain hôn nhân gia đình
→ mở rộng graph neighbors
→ clean HTML giữ Điều/Khoản
→ legal chunking
→ Chroma index
→ BM25 index
→ indexed_manifest.json
```

## Sinh test case

```bash
python scripts/generate_test_cases.py
```

## Benchmark retrieval

```bash
python scripts/run_benchmark.py
# hoặc
python test_evaluation.py
```

## Test retrieval nhanh

```bash
python test.py
```

## Chạy API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Request mẫu:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Điều kiện kết hôn theo Luật Hôn nhân và gia đình 2014 là gì?","k":5,"strategy":"auto","generate":false}'
```

Để sinh answer, đặt `generate=true` và cấu hình API key.

## Lưu ý quan trọng

- Sau khi đổi cleaner/chunker/retrieval, nên xóa index cũ hoặc để `reset_on_ingest: true` rồi chạy lại `python scripts/ingest.py`.
- Precision@5 thấp có thể bình thường nếu mỗi câu hỏi chỉ có 1 `relevant_id`, vì trần thực tế khi trả 5 docs là `1/5 = 0.2`.
- Với pháp luật, không chỉ cần retrieve đúng nội dung mà còn cần metadata hiệu lực và quan hệ thay thế/sửa đổi.

## Local PDF corpus: Luật Hôn nhân & Gia đình

Repo đã hỗ trợ thêm corpus hẹp từ PDF chính thức. Cấu hình mặc định trong `configs/config.yaml` đang dùng:

```yaml
dataset:
  source: "local_pdf"
```

Chuẩn bị dữ liệu:

```powershell
copy data\raw\family_law\manifest.example.jsonl data\raw\family_law\manifest.jsonl
copy data\raw\family_law\relationships.example.jsonl data\raw\family_law\relationships.jsonl
```

Sau đó tải các PDF chính thức, đặt vào:

```text
data/raw/family_law/pdfs/
```

Tên PDF phải khớp trường `filename` trong `manifest.jsonl`.

Chạy pipeline:

```powershell
python scripts/ingest_family_law_pdfs.py
python scripts/validate_family_law_corpus.py
```

Pipeline sẽ sinh:

```text
data/processed/family_law/metadata.jsonl
data/processed/family_law/content.jsonl
data/processed/family_law/relationships.jsonl
data/processed/family_law/documents.jsonl
data/processed/family_law/chunks.jsonl
data/chroma/family_law/
data/bm25/family_law_bm25.pkl
data/graph/family_law_relationships.pkl
```

Muốn quay lại baseline HuggingFace thì đổi:

```yaml
dataset:
  source: "huggingface"
```
