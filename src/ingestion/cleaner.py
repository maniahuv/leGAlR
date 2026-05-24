import re
import unicodedata
from html import unescape
from concurrent.futures import ProcessPoolExecutor

from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_core.documents import Document

def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html,"html.parser") #biến html thành cây dom
    for tag in soup(["script", "style"]): #xóa tag không chứa nội dung
        tag.decompose() #xóa khỏi cây dom 
    text = soup.get_text(separator=" ") #nối toàn bộ node của cây dom với nhau bằng dấu cách 
    return unescape(text) #mã html crawl từ web có thể chứa các kí tự escape như &, space -> cần unescape 

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text) #chuẩn hóa về chuẩn thống nhất NFC vì máy có thể nhìn 2 từ mà người thấy giống nhau mà máy thấy khác nhau
    text = text.replace("\xa0", " ") #vẫn phải replace vì có khả năng text đó không phải html nên không chạy _strip_html() 
    text = re.sub(r"\s+", " ", text) #gộp nhiều khoảng trắng thành 1 
    return text.strip() #xóa khoảng trắng đầu và cuối 

# nhận 1 doc 
def clean_document(doc: Document) -> Document: 
    raw = doc.page_content or "" #lấy nội dung chính doc
    if "<" in raw and ">" in raw:
        raw=_strip_html(raw) #nếu trong nội dung có < > thì khả năng là file html, cần gọi _strip_html() để bỏ tag html
    cleaned=_normalize(raw) #chuẩn hóa text raw, bao gồm chuẩn hóa unicode, xóa space, xóa xuống dòng dư
    return Document(
        page_content=cleaned,
        metadata=doc.metadata
    )

# nhận n docs 
def clean_documents(docs: list[Document], workers: int = 1) -> list[Document]: #workers là số tiến trình xử lý //
    if workers <= 1:
        return [
            clean_document(d) 
            for d in tqdm(docs, desc="Cleaning", unit="doc")
        ] #tqdm là thanh tiến trình
    with ProcessPoolExecutor(max_workers=workers) as executor: #tạo nhóm pool các process chạy song song
        return list(
            tqdm(
                executor.map(clean_document, docs, chunksize=64), #chạy song song, mỗi worker xử lý 1 batch 64 docs 1 lần, số lượng wokers quy định ở trong hàm
                total=len(docs), desc="Cleaning", unit="doc",
            )
        )
