from __future__ import annotations

import sys
import types
import pandas as pd
from typing import Any, Dict, List
from datasets import Dataset

# --- BẮT ĐẦU VÁ LỖI (MONKEY PATCH) ---
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module
# --- KẾT THÚC VÁ LỖI ---

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# Thêm bộ nhúng Google GenAI Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.llm import get_llm

class LegalRagasEvaluator:
    def __init__(self, metrics: list | None = None):
        self.llm = get_llm()
        
        # CHỐNG LỖI OPENAI: Khai báo rõ dùng Google Embeddings cho Ragas
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        self.metrics = metrics or [
            context_precision, 
            context_recall,    
            faithfulness,      
            answer_relevancy,  
        ]

    def _prepare_hf_dataset(self, local_results: List[Dict[str, Any]]) -> Dataset:
        data = {"question": [], "contexts": [], "answer": [], "ground_truth": []}
        for item in local_results:
            data["question"].append(item.get("question", ""))
            
            raw_contexts = item.get("retrieved_contexts", [])
            contexts_str = []
            for ctx in raw_contexts:
                if hasattr(ctx, "page_content"):
                    contexts_str.append(ctx.page_content)
                elif isinstance(ctx, str):
                    contexts_str.append(ctx)
                else:
                    contexts_str.append(str(ctx))
                    
            data["contexts"].append(contexts_str)
            data["answer"].append(item.get("generated_answer", ""))
            data["ground_truth"].append(item.get("ground_truth", ""))
            
        return Dataset.from_dict(data)

    def evaluate_results(self, local_results: List[Dict[str, Any]]) -> pd.DataFrame:
        if not local_results:
            print("⚠️ Không có dữ liệu để đánh giá.")
            return pd.DataFrame()

        print(f"🔄 Đang chuẩn bị {len(local_results)} test cases cho RAGAS...")
        hf_dataset = self._prepare_hf_dataset(local_results)
        print("🚀 Bắt đầu chạy RAGAS Evaluation...")
        
        # CHỐNG RATE LIMIT CHO BƯỚC 2:
        # Ép Ragas chỉ chạy 1 luồng (max_workers=1) và nếu bị lỗi API sẽ tự chờ 60s
        safe_run_config = RunConfig(max_workers=1, max_retries=10, max_wait=60)
        
        try:
            result = evaluate(
                dataset=hf_dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings, # <-- Bắt buộc phải có dòng này
                run_config=safe_run_config,
                raise_exceptions=False 
            )
            
            df_result = result.to_pandas()
            print("✅ Đánh giá RAGAS hoàn tất!")
            return df_result
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình chạy RAGAS: {e}")
            return pd.DataFrame()