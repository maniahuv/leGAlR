from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.setting import config
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.pdf_loader import (
    load_local_family_law_corpus,
    read_jsonl,
    resolve_repo_path,
    sanitize_metadata,
    write_json,
    write_jsonl,
)
from src.indexing.bm25_index import build_bm25_index, save_bm25_index
from src.indexing.chroma_store import build_chroma_index
from src.retrieval.graph import build_graph


def _doc_to_record(doc) -> dict[str, Any]:
    return {
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata or {}),
    }


def _chunk_to_record(chunk) -> dict[str, Any]:
    meta = dict(chunk.metadata or {})
    return {
        "chunk_uid": meta.get("chunk_uid", ""),
        "doc_id": meta.get("doc_id", ""),
        "title": meta.get("title", ""),
        "so_ky_hieu": meta.get("so_ky_hieu", ""),
        "article": meta.get("article", ""),
        "article_title": meta.get("article_title", ""),
        "clause": meta.get("clause", ""),
        "content": chunk.page_content,
        "metadata": sanitize_metadata(meta),
    }


def _copy_relationships(config, processed_dir: Path) -> list[dict[str, Any]]:
    dataset_cfg = config.dataset
    raw_relationships_path = resolve_repo_path(getattr(dataset_cfg, "relationships_path", "data/raw/family_law/relationships.jsonl"))
    relationships = read_jsonl(raw_relationships_path)

    # Normalize minimal schema for GraphRAG compatibility.
    normalized: list[dict[str, Any]] = []
    for row in relationships:
        src = str(row.get("doc_id", "")).strip()
        dst = str(row.get("other_doc_id", "")).strip()
        rel = str(row.get("relationship", "")).strip()
        if src and dst and rel:
            normalized.append({"doc_id": src, "other_doc_id": dst, "relationship": rel})

    write_jsonl(processed_dir / "relationships.jsonl", normalized)
    return normalized


def _save_graph_cache(relationships: list[dict[str, Any]]) -> None:
    graph_cfg = getattr(config, "graph", None)
    graph_path = resolve_repo_path(getattr(graph_cfg, "persist_path", "data/graph/family_law_relationships.pkl"))
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    # Cache the raw relationship list, because retrieval_tools._get_graph builds nx graph from it.
    graph_path.write_bytes(pickle.dumps(relationships))


def _write_index_manifest(chunks, path: Path) -> None:
    doc_ids = sorted({str((c.metadata or {}).get("doc_id", "")).strip() for c in chunks if (c.metadata or {}).get("doc_id")})
    articles = sorted({
        f"{(c.metadata or {}).get('doc_id', '')}::Điều {(c.metadata or {}).get('article', '')}"
        for c in chunks
        if (c.metadata or {}).get("doc_id") and (c.metadata or {}).get("article")
    })
    payload = {
        "indexed_doc_count": len(doc_ids),
        "indexed_chunk_count": len(chunks),
        "indexed_doc_ids": doc_ids,
        "indexed_articles_count": len(articles),
        "indexed_articles_sample": articles[:50],
    }
    write_json(path, payload)


def main() -> None:
    print("🚀 Loading official family-law source files from local manifest...")
    corpus = load_local_family_law_corpus(config)
    if not corpus.documents:
        raise ValueError("No local source documents loaded. Check manifest_path and raw_dir in configs/config.yaml.")

    dataset_cfg = config.dataset
    processed_dir = resolve_repo_path(getattr(dataset_cfg, "processed_dir", "data/processed/family_law"))
    interim_dir = resolve_repo_path(getattr(dataset_cfg, "interim_dir", "data/interim/family_law"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    (interim_dir / "extracted_texts").mkdir(parents=True, exist_ok=True)

    print("💾 Writing HF-style local metadata/content files...")
    write_jsonl(processed_dir / "metadata.jsonl", corpus.metadata_records)
    write_jsonl(processed_dir / "content.jsonl", corpus.content_records)
    write_json(interim_dir / "extraction_report.json", {"files": corpus.extraction_reports})

    for content in corpus.content_records:
        doc_id = str(content.get("doc_id", "unknown"))
        (interim_dir / "extracted_texts" / f"{doc_id}.txt").write_text(
            str(content.get("content_text", "")), encoding="utf-8"
        )

    relationships = _copy_relationships(config, processed_dir)
    _save_graph_cache(relationships)

    print("🧹 Cleaning extracted text while preserving legal structure...")
    cleaned_docs = clean_documents(corpus.documents)
    write_jsonl(processed_dir / "documents.jsonl", [_doc_to_record(d) for d in cleaned_docs])

    print("✂️ Chunking by Article/Clause legal structure...")
    chunks = chunk_documents(cleaned_docs, config)
    if not chunks:
        raise ValueError("No chunks created. Check PDF extraction and chunking config.")
    write_jsonl(processed_dir / "chunks.jsonl", [_chunk_to_record(c) for c in chunks])

    print(f"Loaded source documents: {len(corpus.documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Relationships: {len(relationships)}")

    print("📦 Building Chroma vector index...")
    build_chroma_index(chunks, reset=getattr(config.vector_store, "reset_on_ingest", True))

    print("📦 Building Vietnamese BM25 index...")
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    _write_index_manifest(chunks, ROOT_DIR / "data" / "evaluation" / "indexed_manifest.json")

    report = {
        "source": "local_pdf",
        "documents": len(corpus.documents),
        "chunks": len(chunks),
        "relationships": len(relationships),
        "processed_dir": str(processed_dir),
        "vector_store": getattr(config.vector_store, "persist_directory", "data/chroma/family_law"),
        "bm25_path": getattr(config.bm25, "persist_path", "data/bm25/family_law_bm25.pkl"),
    }
    write_json(processed_dir / "ingest_report.json", report)

    print("✅ LOCAL FAMILY-LAW SOURCE INGEST DONE")
    print("Next: python scripts/validate_family_law_corpus.py")
    print("Then: python scripts/run_benchmark.py")


if __name__ == "__main__":
    main()
