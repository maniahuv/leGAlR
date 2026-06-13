from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


class AgentToolInput(BaseModel):
    query: str = Field(..., description="User request or question.")


class DeepAgentWrapper:
    def __init__(self, graph, name: str = "deep_agent"):
        self.graph = graph
        self.name = name

    def invoke(self, query: str | dict) -> str:
        inputs = {"messages": [HumanMessage(content=query)]} if isinstance(query, str) else query
        result = self.graph.invoke(inputs)
        return self._extract_final_answer(result)

    def stream(self, query: str | dict):
        inputs = {"messages": [HumanMessage(content=query)]} if isinstance(query, str) else query
        return self.graph.stream(inputs)

    def as_tool(self, name: str, description: str) -> StructuredTool:
        def _run(query: str) -> str:
            return self.invoke(query)
        return StructuredTool.from_function(func=_run, name=name, description=description, args_schema=AgentToolInput)

    @staticmethod
    def _extract_final_answer(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if not messages:
                return ""
            last = messages[-1]
            if isinstance(last, AIMessage):
                return last.content
            if hasattr(last, "content"):
                return last.content
            return str(last)
        return str(result)


def create_deep_agent(model, tools: list, system_prompt: str) -> DeepAgentWrapper:
    try:
        graph = create_react_agent(model=model, tools=tools, prompt=system_prompt)
    except TypeError:
        graph = create_react_agent(model=model, tools=tools, state_modifier=system_prompt)
    return DeepAgentWrapper(graph)
