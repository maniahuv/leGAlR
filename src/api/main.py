from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers.query import router as query_router

app = FastAPI(title="Vietnamese Legal RAG API", version="0.3.0")
app.include_router(query_router)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


@app.on_event("startup")
def warmup_retrieval() -> None:
    """
    Warm-up retrieval khi server khởi động.

    Mục đích:
    - Load embedding model trước.
    - Load Chroma/vector store trước.
    - Tránh request đầu tiên trên web bị chậm do cold-start.
    """
    try:
        from src.tools.retrieval_tools import retrieve_documents

        retrieve_documents(
            "Điều kiện thuận tình ly hôn là gì?",
            k=1,
            strategy="dense",
        )

        print("✅ Retrieval warm-up done")
    except Exception as e:
        print(f"⚠️ Retrieval warm-up failed: {e}")


@app.get("/", include_in_schema=False)
def index():
    if WEB_DIR.exists():
        return RedirectResponse(url="/web/index.html")
    return {"status": "ok", "message": "Web UI directory not found."}


@app.get("/health")
def health():
    return {"status": "ok"}