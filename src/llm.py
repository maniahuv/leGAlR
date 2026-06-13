from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.setting import config

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY/GEMINI_API_KEY. Create .env from .env.example or set environment variable.")
    
    kwargs = {
        "model": os.getenv("LLM_MODEL", config.llm.model),
        "temperature": float(os.getenv("LLM_TEMPERATURE", config.llm.temperature)),
        "google_api_key": api_key,
        "streaming": True,
    }

    max_output_tokens = os.getenv("LLM_MAX_OUTPUT_TOKENS") or getattr(config.llm, "max_tokens", None)
    if max_output_tokens:
        kwargs["max_output_tokens"] = int(max_output_tokens)

    return ChatGoogleGenerativeAI(**kwargs)