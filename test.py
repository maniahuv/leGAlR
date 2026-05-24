# from datasets import load_dataset
# dataset=load_dataset("th1nhng0/vietnamese-legal-documents", name="content", split="data")
# print(dataset)

# from tqdm import tqdm
# import time
# # for item in tqdm(iterable): tức là bọc iterable bằng tqdm để theo dõi tiến độ 

# for i in tqdm(range(5), desc="Processing", unit="step"):
#     time.sleep(1)

# import networkx as nx
# import matplotlib.pyplot as plt 

# G=nx.DiGraph()

# triples=[
#     ("Hôn nhân", "có điều kiện", "Đủ tuổi kết hôn"),
#     ("Hôn nhân", "không được vi phạm", "Cấm kết hôn giả tạo"),
#     ("Ly hôn", "được giải quyết bởi", "Tòa án"),
#     ("Tài sản chung", "được chia khi", "Ly hôn"),
#     ("Con chung", "được xem xét khi", "Ly hôn"),
# ]

# for subject, relation, object_ in triples:
#     G.add_node(subject)
#     G.add_node(object_)
#     G.add_edge(subject, object_, relation=relation) # đặt tên cho key là relation 

# pos = nx.spring_layout(G,seed=42)
# # định nghĩa layout hiển thị

# plt.figure(figsize=(12,7))
# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_size=3000,
#     font_size=10,
#     arrows=True
# )

# edge_labels=nx.get_edge_attributes(G, "relation")
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# plt.show()

# entity = "Ly hôn"

# print("Các quan hệ đi ra từ: ", entity)
# for neighbor in G.successors(entity):
#     relation = G[entity][neighbor]["relation"]
#     print(f"{entity} --{relation} --> {neighbor}")

# from src.indexing.chroma_store import get_store
# from src.indexing.bm25_index import load_bm25_index
# from src.retrieval.hybrid import hybrid_search
# from src.retrieval.reranker import rerank

# store = get_store()
# bm25 = load_bm25_index()

# query = "quy định về kết hôn trái pháp luật"

# docs = hybrid_search(store, bm25, query, k=10)
# docs = rerank(query, docs, k=5)

# for i, doc in enumerate(docs, 1):
#     print("=" * 80)
#     print(i)
#     print(doc.page_content[:500])
#     print(doc.metadata)

# import sys
# from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parent
# sys.path.insert(0, str(ROOT_DIR))

# from src.agents.orchestrator import agent


# question = "Quy định về kết hôn trái pháp luật là gì?"

# answer = agent.invoke(question)

# print(answer)

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.orchestrator import legal_rag_agent

question = "Quy định về kết hôn trái pháp luật là gì?"

answer = legal_rag_agent.invoke(question)

print(answer)