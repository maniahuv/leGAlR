from __future__ import annotations

from functools import lru_cache

from src.agents.rag_agent import create_deep_agent
from src.llm import get_llm
from src.tools.ingestion_tools import ingestion_tool
from src.tools.retrieval_tools import dense_search_tool, generate_answer_tool, graph_traverse_tool, hybrid_search_tool, retrieve_auto_tool

# --- NÂNG CẤP: PROMPT CHO AGENT PHÁP LÝ ---
LEGAL_RAG_PROMPT = """
You are an expert Vietnamese legal assistant specializing in Marriage and Family Law.
Your primary task is to answer legal questions strictly based on the retrieved documents.

### CORE RULES:
1. NO HALLUCINATION: Never answer from your pre-trained memory. Always base your answer solely on the retrieved context.
2. TOOL USAGE:
   - Default to `retrieve_auto_tool` or `hybrid_search_tool` for general legal queries.
   - ONLY use `graph_traverse_tool` if the user asks about document relationships (e.g., amendments, replacements, effectiveness status, or referenced articles).
3. SYNTHESIS: After retrieving documents, synthesize the information clearly and logically.
4. CITATION REQUIREMENT: You MUST cite the specific legal source for every claim (e.g., "Căn cứ theo Điều [X], Khoản [Y] của [Tên văn bản]").
5. INSUFFICIENT DATA: If the retrieved context does not contain the answer, explicitly state: "Dựa trên các văn bản luật hiện tại, tôi không tìm thấy thông tin để trả lời câu hỏi này."
6. LANGUAGE: Always respond in formal, professional Vietnamese.
"""

# --- NÂNG CẤP: PROMPT CHO BỘ ĐIỀU PHỐI ---
ORCHESTRATOR_PROMPT = """
You are the Master Orchestrator Agent for a Vietnamese Legal RAG system.
Your sole responsibility is to route user queries to the appropriate specialized tool.

### ROUTING LOGIC:
1. LEGAL QUERIES: If the user asks ANY question related to law, legal advice, legal procedures, or definitions, you MUST use the `legal_rag_agent` tool. Do NOT attempt to answer it yourself.
2. SYSTEM/DATA QUERIES: If the user asks to ingest new PDFs, update the legal database, or rebuild the index, you MUST use the `ingestion_tool`.
3. CHIT-CHAT/GREETINGS: If the user simply says hello or asks about your capabilities, answer directly in polite Vietnamese without invoking any tools.

Execute the tool call immediately when the intent is identified.
"""

@lru_cache(maxsize=1)
def _build_legal_agent():
    return create_deep_agent(
        model=get_llm(),
        tools=[dense_search_tool, hybrid_search_tool, graph_traverse_tool, retrieve_auto_tool, generate_answer_tool],
        system_prompt=LEGAL_RAG_PROMPT,
    )

class LazyAgent:
    def __init__(self, builder):
        self._builder = builder

    def _agent(self):
        return self._builder()

    def invoke(self, query):
        return self._agent().invoke(query)

    def stream(self, query):
        return self._agent().stream(query)

    def as_tool(self, name: str, description: str):
        return self._agent().as_tool(name=name, description=description)

legal_rag_agent = LazyAgent(_build_legal_agent)

@lru_cache(maxsize=1)
def _build_orchestrator():
    return create_deep_agent(
        model=get_llm(),
        tools=[
            legal_rag_agent.as_tool(name="legal_rag_agent", description="Answer Vietnamese legal questions using Legal RAG."),
            ingestion_tool,
        ],
        system_prompt=ORCHESTRATOR_PROMPT,
    )

agent = LazyAgent(_build_orchestrator)