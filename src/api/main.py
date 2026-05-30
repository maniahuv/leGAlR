from __future__ import annotations

from fastapi import FastAPI

from src.api.routers.query import router as query_router

app = FastAPI(title="Vietnamese Legal RAG API", version="0.2.0")
app.include_router(query_router)


@app.get("/health")
def health():
    return {"status": "ok"}
