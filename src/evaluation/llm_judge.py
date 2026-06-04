import json
from src.llm import get_llm


JUDGE_PROMPT = """
You are an evaluator for a Vietnamese legal RAG system.

Evaluate the answer based on the question and retrieved context.

Return JSON only with these fields:
- groundedness: score from 1 to 5
- relevance: score from 1 to 5
- completeness: score from 1 to 5
- legal_safety: score from 1 to 5
- comment: short explanation in Vietnamese

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}
"""


def judge_answer(
    question: str,
    context: str,
    answer: str,
) -> dict:
    """
    Dùng LLM-as-a-judge để chấm answer.

    groundedness: answer có bám context không
    relevance: có trả lời đúng câu hỏi không
    completeness: có đủ ý không
    legal_safety: có cảnh báo/diễn đạt an toàn pháp lý không
    """
    try:
        llm = get_llm()

        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
        )

        response = llm.invoke(prompt)
        content = response.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "groundedness": None,
                "relevance": None,
                "completeness": None,
                "legal_safety": None,
                "comment": content,
                "raw": content,
            }

    except Exception as e:
        return {
            "groundedness": None,
            "relevance": None,
            "completeness": None,
            "legal_safety": None,
            "comment": "LLM judge failed.",
            "error": str(e),
        }