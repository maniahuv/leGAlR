from typing import Any
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.prebuilt import create_react_agent


class AgentToolInput(BaseModel):
    query: str = Field(
        ...,
        description="User request or question that should be handled by this agent."
    )


class DeepAgentWrapper:
    """
    Wrapper cho LangGraph agent.

    Mục tiêu:
    - Gọi agent bằng .invoke("câu hỏi")
    - Convert agent thành tool bằng .as_tool(...)
    """

    def __init__(self, graph, name: str = "deep_agent"):
        self.graph = graph
        self.name = name

    def invoke(self, query: str | dict) -> str:
        """
        Invoke agent và trả về final text.
        """
        if isinstance(query, str):
            inputs = {
                "messages": [
                    HumanMessage(content=query)
                ]
            }
        else:
            inputs = query

        result = self.graph.invoke(inputs)

        return self._extract_final_answer(result)

    def stream(self, query: str | dict):
        """
        Stream agent nếu cần debug.
        """
        if isinstance(query, str):
            inputs = {
                "messages": [
                    HumanMessage(content=query)
                ]
            }
        else:
            inputs = query

        return self.graph.stream(inputs)

    def as_tool(self, name: str, description: str) -> StructuredTool:
        """
        Convert agent thành tool để agent khác gọi.
        """

        def _run(query: str) -> str:
            return self.invoke(query)

        return StructuredTool.from_function(
            func=_run,
            name=name,
            description=description,
            args_schema=AgentToolInput,
        )

    @staticmethod
    def _extract_final_answer(result: Any) -> str:
        """
        Lấy nội dung trả lời cuối cùng từ LangGraph result.
        """
        if isinstance(result, str):
            return result

        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]

            if not messages:
                return ""

            last_message = messages[-1]

            if isinstance(last_message, AIMessage):
                return last_message.content

            if hasattr(last_message, "content"):
                return last_message.content

            return str(last_message)

        return str(result)


def create_deep_agent(model, tools: list, system_prompt: str) -> DeepAgentWrapper:
    """
    Tạo ReAct agent bằng LangGraph.

    model: LLM object
    tools: list LangChain tools
    system_prompt: system instruction
    """

    try:
        graph = create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt,
        )
    except TypeError:
        # Một số version LangGraph cũ dùng state_modifier thay vì prompt
        graph = create_react_agent(
            model=model,
            tools=tools,
            state_modifier=system_prompt,
        )

    return DeepAgentWrapper(graph)