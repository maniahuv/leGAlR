from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(config) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=[
            "\nChương ",
            "\nMục ",
            "\nĐiều ",
            "\nKhoản ",
            "\nĐiểm ",
            "\n\n",
            "\n",
            ". ",
            "; ",
            " ",
            "",
        ],
        chunk_size=int(config.chunking.chunk_size),
        chunk_overlap=int(config.chunking.chunk_overlap),
    )


def _extract_article(text: str) -> str:
    m = re.search(r"\bĐiều\s+(\d+[a-zA-Z]?)\s*[\.:]", text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _extract_clause(text: str) -> str:
    m = re.search(r"\bKhoản\s+(\d+)\b", text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _split_by_article(doc: Document, config) -> list[Document]:
    text = doc.page_content or ""
    if not re.search(r"\bĐiều\s+\d+[a-zA-Z]?\s*[\.:]", text, flags=re.IGNORECASE):
        return build_splitter(config).split_documents([doc])

    parts = re.split(r"(?=\n?Điều\s+\d+[a-zA-Z]?\s*[\.:])", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]

    splitter = build_splitter(config)
    output: list[Document] = []
    max_len = int(config.chunking.chunk_size) + int(config.chunking.chunk_overlap)
    for part in parts:
        part_doc = Document(page_content=part, metadata=dict(doc.metadata or {}))
        if len(part) > max_len:
            output.extend(splitter.split_documents([part_doc]))
        else:
            output.append(part_doc)
    return output


def _add_chunk_metadata(chunks: list[Document]) -> list[Document]:
    counters = defaultdict(int)
    for chunk in chunks:
        meta = dict(chunk.metadata or {})
        doc_id = str(meta.get("doc_id") or meta.get("source") or "unknown").strip()
        idx = counters[doc_id]
        counters[doc_id] += 1
        meta["doc_id"] = doc_id
        meta["chunk_index"] = idx
        meta["chunk_uid"] = f"{doc_id}_{idx}"
        article = _extract_article(chunk.page_content)
        clause = _extract_clause(chunk.page_content)
        if article:
            meta["article"] = article
        if clause:
            meta["clause"] = clause
        chunk.metadata = meta
    return chunks


def chunk_documents(docs: list[Document], config) -> list[Document]:
    strategy = str(getattr(config.chunking, "strategy", "legal_recursive"))
    chunks: list[Document] = []
    if strategy == "legal_recursive":
        for doc in docs:
            chunks.extend(_split_by_article(doc, config))
    else:
        chunks = build_splitter(config).split_documents(docs)
    return _add_chunk_metadata(chunks)
