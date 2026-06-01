from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from html import unescape

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from tqdm import tqdm


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # separator="\n" giúp giữ lại ranh giới điều/khoản/điểm tốt hơn separator=" ".
    text = soup.get_text(separator="\n")
    return unescape(text)


def _restore_legal_breaks(text: str) -> str:
    """Đưa các mốc pháp lý về đầu dòng để chunker không cắt gãy Điều/Khoản/Điểm."""
    patterns = [
        r"(?<!\n)(Chương\s+[IVXLCDM]+\b)",
        r"(?<!\n)(Mục\s+\d+\b)",
        r"(?<!\n)(Điều\s+\d+[a-zA-Z]?\s*[\.:])",
        r"(?<!\n)(Khoản\s+\d+\b)",
        r"(?<!\n)(Điểm\s+[a-z]\b)",
    ]
    for pattern in patterns:
        text = re.sub(pattern, r"\n\1", text, flags=re.IGNORECASE)
    return text


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\xa0", " ")
    text = _restore_legal_breaks(text)
    # Gộp khoảng trắng ngang, nhưng KHÔNG xóa toàn bộ xuống dòng.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document(doc: Document) -> Document:
    raw = doc.page_content or ""
    if "<" in raw and ">" in raw:
        raw = _strip_html(raw)
    cleaned = _normalize(raw)
    metadata = dict(doc.metadata or {})
    return Document(page_content=cleaned, metadata=metadata)


def clean_documents(docs: list[Document], workers: int = 1) -> list[Document]:
    if workers <= 1:
        return [clean_document(d) for d in tqdm(docs, desc="Cleaning", unit="doc")]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(executor.map(clean_document, docs, chunksize=64), total=len(docs), desc="Cleaning", unit="doc"))
