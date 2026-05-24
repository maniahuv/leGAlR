import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.benchmark import run_retrieval_benchmark

def main():
    # Nhập hoặc đọc trực tiếp chuỗi JSON bộ 100 câu mới của bạn
    test_cases_path = ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json"
    
    # Đoạn phòng vệ nếu chưa có file cứng, tự động tạo thư mục và ghi file dữ liệu mới
    test_cases_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Bạn hãy đảm bảo nội dung JSON mới đã được lưu vào đường dẫn này
    # Nếu file chưa tồn tại, đoạn code dưới đây mẫu cấu trúc để nạp dữ liệu từ chuỗi bạn cung cấp
    if not test_cases_path.exists():
        raw_cases = [
            # Paste mảng 100 câu hỏi mới của bạn vào đây hoặc lưu trực tiếp thành file JSON
        ]
        with open(test_cases_path, "w", encoding="utf-8") as f:
            json.dump(raw_cases, f, ensure_ascii=False, indent=2)

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    strategies = ["dense", "hybrid", "hybrid_rerank", "graph"]

    print("=" * 90)
    print(f"🚀 CHẠY THỰC NGHIỆM VỚI BỘ GOLD DATASET CHUẨN HOÁ ({len(test_cases)} CÂU HỎI)")
    print("=" * 90)

    for strategy in strategies:
        print("\n" + "-"*40)
        print(f"Stratgey: {strategy.upper()}")
        print("-" * 40)

        result = run_retrieval_benchmark(
            test_cases=test_cases,
            strategy=strategy,
            k=5,
        )

        # 1. In kết quả tổng hợp toàn bộ (Global Metrics)
        print("\n[GLOBAL OVERALL METRICS]")
        print(f" -> Hit@5:       {result['summary']['hit_at_k']:.4f}")
        print(f" -> Precision@5: {result['summary']['precision_at_k']:.4f}")
        print(f" -> Recall@5:    {result['summary']['recall_at_k']:.4f}")
        print(f" -> MRR:         {result['summary']['reciprocal_rank']:.4f}")

        # 2. Bóc tách chi tiết hiệu năng theo từng bài toán (Breakdown Scenarios)
        print("\n[BREAKDOWN BY SCENARIOS]")
        for scenario, metrics in result["breakdown"].items():
            if scenario == "global_overall":
                continue
            print(f" 📂 Phân đoạn: {scenario}")
            print(f"    * Hit@5: {metrics['hit_at_k']:.3f} | Precision@5: {metrics['precision_at_k']:.3f} | Recall@5: {metrics['recall_at_k']:.3f} | MRR: {metrics['reciprocal_rank']:.3f}")

if __name__ == "__main__":
    main()