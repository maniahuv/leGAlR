from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.legal_parser import (
    extract_article_number,
    extract_article_title,
    extract_first_clause_number,
    legal_chunk_uid,
    split_text_by_articles,
)


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


def _split_by_article(doc: Document, config) -> list[Document]:
    """Prefer one Article per chunk for legal grounding and ArticleHit@k."""
    text = doc.page_content or ""
    articles = split_text_by_articles(text)
    if not articles:
        return build_splitter(config).split_documents([doc])

    splitter = build_splitter(config)
    output: list[Document] = []
    max_len = int(config.chunking.chunk_size) + int(config.chunking.chunk_overlap)

    for article in articles:
        article_meta = dict(doc.metadata or {})
        if article.article:
            article_meta["article"] = article.article
        if article.title:
            article_meta["article_title"] = article.title

        source_header = " | ".join(
            x
            for x in [
                str(article_meta.get("title", "")).strip(),
                f"Số hiệu: {article_meta.get('so_ky_hieu', '')}" if article_meta.get("so_ky_hieu") else "",
                f"Hiệu lực: {article_meta.get('tinh_trang_hieu_luc', '')}" if article_meta.get("tinh_trang_hieu_luc") else "",
            ]
            if x
        )
        page_content = f"{source_header}\n{article.text}" if source_header else article.text
        part_doc = Document(page_content=page_content, metadata=article_meta)
        if len(article.text) > max_len:
            split_parts = splitter.split_documents([part_doc])
            for split_part in split_parts:
                split_meta = dict(split_part.metadata or {})
                split_meta.setdefault("article", article.article)
                split_meta.setdefault("article_title", article.title)
                split_part.metadata = split_meta
            output.extend(split_parts)
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

        article = str(meta.get("article") or extract_article_number(chunk.page_content)).strip()
        article_title = str(meta.get("article_title") or extract_article_title(chunk.page_content)).strip()
        clause = str(meta.get("clause") or extract_first_clause_number(chunk.page_content)).strip()

        meta["doc_id"] = doc_id
        meta["chunk_index"] = idx
        meta["chunk_uid"] = legal_chunk_uid(doc_id, article, idx)
        if article:
            meta["article"] = article
        if article_title:
            meta["article_title"] = article_title
        if clause:
            meta["clause"] = clause
        chunk.metadata = meta
    return chunks


def chunk_documents(docs: list[Document], config) -> list[Document]:
    strategy = str(getattr(config.chunking, "strategy", "legal_recursive"))
    chunks: list[Document] = []
    if strategy in {"legal_recursive", "legal_article"}:
        for doc in docs:
            chunks.extend(_split_by_article(doc, config))
    else:
        chunks = build_splitter(config).split_documents(docs)
    return _add_chunk_metadata(chunks)
