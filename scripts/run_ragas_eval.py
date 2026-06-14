import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.tools.retrieval_tools import retrieve_documents
from src.llm import get_llm
from src.evaluation.ragas_evaluator import LegalRagasEvaluator
from langchain_core.messages import HumanMessage, SystemMessage

def parse_args():
    parser = argparse.ArgumentParser(description="Chạy benchmark RAGAS cho Legal RAG")
    parser.add_argument("--test_file", type=str, default="src/evaluation/family_law_test_cases.json", help="Đường dẫn file test cases")
    parser.add_argument("--output_dir", type=str, default="data/processed/evaluation", help="Thư mục lưu kết quả")
    parser.add_argument("--strategy", type=str, default="hybrid", choices=["dense", "hybrid", "graph", "auto"], help="Chiến lược retrieval cần đánh giá")
    parser.add_argument("--k", type=int, default=5, help="Số lượng chunks trả về (top-k)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số test case")
    return parser.parse_args()

def load_test_cases(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    args = parse_args()
    print(f"🚀 RAGAS Evaluation: [{args.strategy.upper()}] | Top-K: {args.k}")
    
    test_cases = load_test_cases(args.test_file)
    if args.limit > 0:
        test_cases = test_cases[:args.limit]
    
    llm = get_llm()
    local_results = []
    
    print("\n🔍 Bước 1: Bắt đầu truy xuất và sinh câu trả lời (Có chế độ chống Rate Limit)...")
    for item in tqdm(test_cases, desc="Processing Test Cases"):
        query = item.get("question", "") or item.get("query", "")
        ground_truth = item.get("ground_truth", "") or item.get("expected_answer", "") or item.get("answer", "")
        
        if not ground_truth:
            rel_ids = item.get("relevant_ids", [])
            rel_arts = item.get("relevant_articles", [])
            if rel_ids or rel_arts:
                arts_str = ", ".join(str(a) for a in rel_arts) if rel_arts else ""
                ids_str = ", ".join(str(i) for i in rel_ids) if rel_ids else ""
                ground_truth = f"Căn cứ theo quy định tại: {arts_str} thuộc văn bản {ids_str}"
        
        if not query or not ground_truth:
            continue
            
        # --- BẮT ĐẦU CƠ CHẾ RETRY CHỐNG LỖI 429 ---
        max_retries = 2
        success = False
        
        for attempt in range(max_retries):
            try:
                docs = retrieve_documents(query, k=args.k, strategy=args.strategy)
                contexts = []
                for doc in docs:
                    if hasattr(doc, "page_content"): contexts.append(doc.page_content)
                    elif isinstance(doc, dict) and "content" in doc: contexts.append(doc["content"])
                    else: contexts.append(str(doc))
                
                context_str = "\n---\n".join(contexts)
                system_prompt = (
                    "Bạn là trợ lý pháp luật Việt Nam chuyên về Hôn nhân và Gia đình. "
                    "Dựa vào các quy định pháp luật được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác và ngắn gọn."
                )
                user_prompt = f"QUY ĐỊNH PHÁP LUẬT:\n{context_str}\n\nCÂU HỎI:\n{query}"
                
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # Trích xuất text an toàn
                if hasattr(response, "content"):
                    if isinstance(response.content, str): answer = response.content
                    elif isinstance(response.content, list):
                        text_parts = [p["text"] for p in response.content if isinstance(p, dict) and "text" in p]
                        answer = " ".join(text_parts) if text_parts else str(response.content)
                    else: answer = str(response.content)
                else: answer = str(response)
                
                local_results.append({
                    "question": query,
                    "retrieved_contexts": contexts,
                    "generated_answer": answer,
                    "ground_truth": ground_truth
                })
                
                success = True
                break # Thoát khỏi vòng lặp retry nếu thành công
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"\n⏳ Quá giới hạn API (Lỗi 429). Hệ thống tự động nghỉ 40s (Lần {attempt+1}/{max_retries})...")
                    time.sleep(40) # Nghỉ xả hơi chờ Google cấp lại quota
                else:
                    print(f"\n⚠️ Lỗi xử lý câu '{query}': {e}")
                    break
                    
        # Bắt buộc nghỉ ngơi 5 giây sau MỖI CÂU HỎI để nuôi luồng (Dưới 12 request/phút)
        if success:
            time.sleep(5)
            
    print(f"\n⚖️ Bước 2: Đưa {len(local_results)} kết quả vào RAGAS Evaluator...")
    evaluator = LegalRagasEvaluator()
    df_result = evaluator.evaluate_results(local_results)
    
    if not df_result.empty:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / f"ragas_results_{args.strategy}_k{args.k}.csv"
        df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
        
        print("\n📊 TỔNG KẾT ĐIỂM SỐ RAGAS (Trung bình):")
        for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            if m in df_result.columns:
                print(f" 🔹 {m.replace('_', ' ').title()}: {df_result[m].mean():.4f}")
        
        print(f"\n💾 Đã lưu báo cáo chi tiết tại: {out_path}")

if __name__ == "__main__":
    main()