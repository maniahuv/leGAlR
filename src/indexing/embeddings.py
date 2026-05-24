from langchain_huggingface import HuggingFaceEmbeddings
from configs.setting import config

def get_embedding_model():
    """
    Khởi tạo embedding model dùng cho vector search.
    """
    return HuggingFaceEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs={"device": config.embedding.device},
        encode_kwargs={"normalize_embeddings":True},
    )

