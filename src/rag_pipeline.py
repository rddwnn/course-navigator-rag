from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from openai import OpenAI

from .config import settings
from .raptor_index import RaptorLiteIndex, SummaryNode, Chunk
from .tokenizer import count_tokens

client = OpenAI()

TOP_K_NODES = 5

MAX_CONTEXT_TOKENS = 1500


@dataclass
class LoadedCourseIndex:
    course_id: str
    index: RaptorLiteIndex

_INDEX_CACHE: Dict[str, LoadedCourseIndex] = {}


def get_course_index(course_id: str) -> RaptorLiteIndex:
    if course_id in _INDEX_CACHE:
        return _INDEX_CACHE[course_id].index

    course_dir = Path("data") / course_id
    index = RaptorLiteIndex.from_disk(course_dir)
    _INDEX_CACHE[course_id] = LoadedCourseIndex(course_id=course_id, index=index)
    return index


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)

    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    m = matrix / m_norms

    sims = m @ q
    return sims

def select_top_nodes_and_chunks(
    course_index: RaptorLiteIndex,
    question: str,
    top_k_nodes: int = TOP_K_NODES,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> List[Chunk]:

    emb_response = client.embeddings.create(
        model=settings.embedding_model,
        input=[question],
    )
    question_vec = np.array(emb_response.data[0].embedding, dtype=np.float32)

    sims = cosine_similarity_matrix(question_vec, course_index.embeddings_level_1)
    top_indices = np.argsort(-sims)[:top_k_nodes]

    nodes: List[SummaryNode] = []
    for node in course_index.nodes_by_id.values():
        nodes.append(node)
    nodes_sorted = sorted(nodes, key=lambda n: n.embedding_index)

    selected_nodes: List[SummaryNode] = []
    for i in top_indices:
        if 0 <= i < len(nodes_sorted):
            selected_nodes.append(nodes_sorted[i])
    selected_chunks: List[Chunk] = []
    seen_chunk_ids = set()
    total_tokens = 0

    for node in selected_nodes:
        for chunk_id in node.children_ids:
            if chunk_id in seen_chunk_ids:
                continue
            chunk = course_index.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            num_tokens = getattr(chunk, "num_tokens", None)
            if num_tokens is None or num_tokens == 0:
                num_tokens = count_tokens(chunk.text, settings.openai_model)

            if total_tokens + num_tokens > max_context_tokens:
                continue

            selected_chunks.append(chunk)
            seen_chunk_ids.add(chunk_id)
            total_tokens += num_tokens

    return selected_chunks


def build_context_from_chunks(chunks: List[Chunk]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        header = f"[{chunk.source_id} | chunk {chunk.position.get('chunk_index', '?')}]"
        parts.append(header)
        parts.append(chunk.text)
        parts.append("")  # пустая строка как разделитель

    return "\n".join(parts)

def generate_answer_with_llm(
    question: str,
    context: str,
) -> str:

    if not context.strip():
        system_prompt = (
            "You are a helpful teaching assistant. "
            "You have no course materials available for this query. "
            "Explain that you cannot answer based on the course data."
        )
        user_content = f"Question: {question}"
    else:
        system_prompt = (
            "You are a helpful teaching assistant for a university course. "
            "Use ONLY the provided course context to answer the question. "
            "If the context is not sufficient, say that you are not sure "
            "instead of making things up.\n\n"
            "Answer in English, briefly and clearly (3-6 sentences)."
        )
        user_content = (
            "COURSE CONTEXT:\n"
            "----------------\n"
            f"{context}\n\n"
            "STUDENT QUESTION:\n"
            "----------------\n"
            f"{question}\n\n"
            "Answer briefly and clearly, using only the context."
        )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def answer_question(course_id: str, question: str) -> str:
    index = get_course_index(course_id)
    chunks = select_top_nodes_and_chunks(index, question)
    context = build_context_from_chunks(chunks)
    answer = generate_answer_with_llm(question, context)
    return answer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python -m course_navigator_rag.rag_pipeline <course_id> \"<question>\"")
        sys.exit(1)

    course_id = sys.argv[1]
    question = " ".join(sys.argv[2:])

    print(f"Course: {course_id}")
    print(f"Question: {question}")
    print("-" * 40)

    answer = answer_question(course_id, question)
    print("Answer:")
    print(answer)
