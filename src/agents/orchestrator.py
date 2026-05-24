from src.llm import get_llm

from src.tools.retrieval_tools import (
    dense_search_tool,
    hybrid_search_tool,
    graph_traverse_tool,
    generate_answer_tool,
)

from src.tools.ingestion_tools import ingestion_tool
from src.agents.rag_agent import create_deep_agent


LEGAL_RAG_PROMPT = """
You are a legal assistant specializing in Vietnamese law.

You can use the following tools:
- dense_search_tool
- hybrid_search_tool
- graph_traverse_tool
- generate_answer_tool

Rules:
- Use graph_traverse_tool for relational questions about amendments, replacements, abolitions, references, and legal hierarchy.
- Use hybrid_search_tool for questions involving dates, effectiveness, document numbers, article numbers, and exact legal keywords.
- Use dense_search_tool for general semantic legal questions.
- ALWAYS use generate_answer_tool to produce the final answer after retrieval.
- Do not answer from memory.
- If retrieved documents are expired, mention that clearly.
"""


legal_rag_agent = create_deep_agent(
    model=get_llm(),
    tools=[
        dense_search_tool,
        hybrid_search_tool,
        graph_traverse_tool,
        generate_answer_tool,
    ],
    system_prompt=LEGAL_RAG_PROMPT,
)


ORCHESTRATOR_PROMPT = """
You are an orchestrator agent.

Available tools:
- legal_rag_agent: answer Vietnamese legal questions using a RAG pipeline
- ingestion_tool: ingest, process, index, or rebuild legal document indexes

Rules:
- If the user asks a Vietnamese legal question, call legal_rag_agent.
- If the user asks to ingest, process, index, rebuild, or update documents, call ingestion_tool.
- Do NOT answer directly. Always call the appropriate tool.
"""


agent = create_deep_agent(
    model=get_llm(),
    tools=[
        legal_rag_agent.as_tool(
            name="legal_rag_agent",
            description="Answer Vietnamese legal questions using dense retrieval, hybrid retrieval, graph traversal, and answer generation.",
        ),
        ingestion_tool,
    ],
    system_prompt=ORCHESTRATOR_PROMPT,
)