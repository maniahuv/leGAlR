from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

ARTICLE_RE = re.compile(r"(?im)(?=^\s*Điều\s+(\d+[a-zA-Z]?)\s*[\.:])")
ARTICLE_HEAD_RE = re.compile(r"(?im)^\s*Điều\s+(\d+[a-zA-Z]?)\s*[\.:]\s*(.*)")
CHAPTER_RE = re.compile(r"(?im)^\s*Chương\s+([IVXLCDM\d]+)\s*\.?\s*$|^\s*Chương\s+([IVXLCDM\d]+)\s*\.?\s+(.+)$")
SECTION_RE = re.compile(r"(?im)^\s*Mục\s+([IVXLCDM\d]+)\s*\.?\s*$|^\s*Mục\s+([IVXLCDM\d]+)\s*\.?\s+(.+)$")
CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)\.\s+")
POINT_RE = re.compile(r"(?m)^\s*([a-zđ])\)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class LegalArticle:
    article: str
    title: str
    text: str
    chapter: str = ""
    section: str = ""


def strip_page_markers(text: str) -> str:
    return re.sub(r"\n?\[PAGE\s+\d+\]\n?", "\n", text or "", flags=re.IGNORECASE)



def _nearest_heading(text: str, end_pos: int, pattern: re.Pattern) -> str:
    """Return the nearest legal hierarchy heading before an article.

    Many official PDFs put the heading code (e.g. ``Chương I``) on one line and
    the title on the next line. This helper preserves both when available.
    """
    window_start = max(0, end_pos - 3000)
    window = text[window_start:end_pos]
    matches = list(pattern.finditer(window))
    if not matches:
        return ""

    m = matches[-1]
    lines = window[m.start():].splitlines()
    if not lines:
        return ""
    first = lines[0].strip()
    # If the title is on the next non-empty line, include it unless it is an
    # article/section/chapter marker.
    title = ""
    for line in lines[1:4]:
        cand = line.strip()
        if not cand:
            continue
        low = cand.lower()
        if low.startswith(("điều ", "mục ", "chương ")):
            break
        title = cand
        break
    return f"{first}: {title}" if title else first

def extract_article_number(text: str) -> str:
    m = ARTICLE_HEAD_RE.search(text or "")
    return m.group(1).strip() if m else ""


def extract_article_title(text: str) -> str:
    m = ARTICLE_HEAD_RE.search(text or "")
    if not m:
        return ""
    return (m.group(2) or "").strip()


def extract_first_clause_number(text: str) -> str:
    m = CLAUSE_RE.search(text or "")
    return m.group(1).strip() if m else ""


def split_text_by_articles(text: str) -> list[LegalArticle]:
    """Split legal text into Article-level blocks while preserving the heading."""
    normalized = strip_page_markers(text)
    matches = list(ARTICLE_RE.finditer(normalized))
    if not matches:
        return []

    articles: list[LegalArticle] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        block = normalized[start:end].strip()
        if not block:
            continue
        article = extract_article_number(block)
        title = extract_article_title(block)
        articles.append(
            LegalArticle(
                article=article,
                title=title,
                text=block,
                chapter=_nearest_heading(normalized, start, CHAPTER_RE),
                section=_nearest_heading(normalized, start, SECTION_RE),
            )
        )
    return articles


def detect_articles(text: str) -> list[str]:
    return [a.article for a in split_text_by_articles(text) if a.article]


def legal_chunk_uid(doc_id: str, article: str, chunk_index: int) -> str:
    safe_article = article or "no_article"
    return f"{doc_id}__article_{safe_article}__chunk_{chunk_index}"
