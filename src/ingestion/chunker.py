from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(config) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=[
            "\nĐiều ",
            "\nKhoản ",
            "\nĐiểm ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )


def chunk_documents(docs: list[Document], config) -> list[Document]:
    splitter = build_splitter(config)
    chunks = splitter.split_documents(docs)

    counters = defaultdict(int)

    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", "unknown"))
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_index"] = counters[doc_id]
        counters[doc_id] += 1

    return chunks