from __future__ import annotations

from functools import lru_cache

from src.agents.rag_agent import create_deep_agent
from src.llm import get_llm
from src.tools.ingestion_tools import ingestion_tool
from src.tools.retrieval_tools import dense_search_tool, generate_answer_tool, graph_traverse_tool, hybrid_search_tool, retrieve_auto_tool

LEGAL_RAG_PROMPT = """
You are a Vietnamese legal RAG assistant.
Rules:
- Do not answer from memory.
- Prefer retrieve_auto_tool. Use graph_traverse_tool for amendments/replacements/effectiveness/reference questions.
- After retrieval, use generate_answer_tool or produce a final answer grounded only in retrieved context.
- Always mention legal source metadata when available.
- If context is insufficient, say so.
"""

ORCHESTRATOR_PROMPT = """
You are an orchestrator agent.
- Legal questions -> legal_rag_agent.
- Ingest/rebuild index requests -> ingestion_tool.
Do not answer directly when a tool is needed.
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
