from typing import Literal

from .config import settings
from openai import OpenAI

client = OpenAI()  # ключ возьмётся из окружения, которое мы загрузили в config.py


CourseId = str


async def answer_question(
    course_id: CourseId,
    question: str,
) -> str:
    """
    Главная точка входа в RAG-пайплайн.
    Пока заглушка. Потом сюда подключим:
    - загрузку индекса по course_id
    - retrieval (RAPTOR-style)
    - вызов LLM с контекстом
    """
    # TODO: заменить на реальный RAG-пайплайн
    return f"[{course_id}] Я пока только заглушка. Вопрос: {question}"
