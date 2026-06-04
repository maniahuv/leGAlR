from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from configs.setting import config


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs={"device": config.embedding.device},
        encode_kwargs={"normalize_embeddings": True},
    )
