import os
import glob
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Trực quan hóa kết quả đánh giá RAGAS")
    parser.add_argument("--input_dir", type=str, default="data/processed/evaluation", help="Thư mục chứa các file CSV kết quả")
    parser.add_argument("--output_dir", type=str, default="web", help="Thư mục lưu biểu đồ HTML (mặc định lưu vào web/ để xem qua UI)")
    return parser.parse_args()

def load_and_aggregate_data(input_dir: str):
    """
    Đọc tất cả các file CSV trong thư mục và tính điểm trung bình.
    """
    path = Path(input_dir)
    csv_files = glob.glob(str(path / "ragas_results_*.csv"))
    
    if not csv_files:
        print(f"⚠️ Không tìm thấy file CSV nào trong {input_dir}.")
        return None

    aggregated_data = {}
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    
    for file in csv_files:
        # Trích xuất tên chiến lược từ tên file (VD: ragas_results_hybrid_k5.csv -> hybrid_k5)
        filename = os.path.basename(file)
        strategy_name = filename.replace("ragas_results_", "").replace(".csv", "").upper()
        
        df = pd.read_csv(file)
        
        # Tính trung bình các cột metric
        scores = []
        for m in metrics:
            if m in df.columns:
                scores.append(df[m].mean())
            else:
                scores.append(0.0) # Fallback nếu thiếu cột
                
        aggregated_data[strategy_name] = scores
        
    return aggregated_data, metrics

def create_visualizations(data: dict, metrics: list, output_dir: str):
    """
    Vẽ 2 biểu đồ: Radar (mạng nhện) để so sánh đa chiều và Bar (cột) để xem tổng quan.
    """
    # Vietsub các nhãn để báo cáo đẹp hơn
    labels = ["Độ chính xác ngữ cảnh<br>(Context Precision)", 
              "Độ phủ ngữ cảnh<br>(Context Recall)", 
              "Độ trung thực<br>(Faithfulness)", 
              "Độ bám sát câu hỏi<br>(Answer Relevancy)"]

    # Khởi tạo layout chứa 2 biểu đồ cạnh nhau (1 Radar, 1 Bar)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "bar"}]],
        subplot_titles=("So sánh Đa chiều (Radar Chart)", "Điểm Trung bình Tổng thể (Bar Chart)")
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (strategy, scores) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        
        # 1. Thêm Radar Chart Trace
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], # Nối điểm cuối về điểm đầu để khép kín đa giác
            theta=labels + [labels[0]],
            fill='toself',
            name=strategy,
            line=dict(color=color),
            marker=dict(size=8)
        ), row=1, col=1)

        # 2. Thêm Bar Chart Trace (Tính trung bình của 4 điểm số)
        overall_score = sum(scores) / len(scores)
        fig.add_trace(go.Bar(
            x=[strategy],
            y=[overall_score],
            name=strategy,
            marker_color=color,
            text=[f"{overall_score:.3f}"],
            textposition='auto',
            showlegend=False # Không cần hiện legend trùng lặp ở Bar chart
        ), row=1, col=2)

    # Căn chỉnh Layout
    fig.update_layout(
        title_text="BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG LEGAL RAG (RAGAS FRAMEWORK)",
        title_x=0.5,
        title_font=dict(size=20, family="Arial", color="black"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12, family="Arial"))
        ),
        yaxis=dict(range=[0, 1], title="Điểm số (0.0 - 1.0)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=600,
        width=1100,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Lưu kết quả ra file HTML
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "ragas_dashboard.html"
    
    fig.write_html(str(html_file))
    print(f"✅ Đã tạo biểu đồ tương tác tại: {html_file}")
    print(f"👉 Mẹo: Bạn có thể mở trực tiếp file này trên trình duyệt hoặc truy cập qua FastAPI: http://localhost:8000/web/ragas_dashboard.html")

def main():
    args = parse_args()
    print("📊 Đang đọc dữ liệu đánh giá RAGAS...")
    result = load_and_aggregate_data(args.input_dir)
    
    if result:
        data, metrics = result
        print(f"Thấy dữ liệu của {len(data)} chiến lược: {', '.join(data.keys())}")
        create_visualizations(data, metrics, args.output_dir)

if __name__ == "__main__":
    main()