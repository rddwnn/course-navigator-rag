from __future__ import annotations


from pathlib import Path
from typing import List, Dict, Any

import json
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

from .config import settings
from .tokenizer import split_into_token_chunks, count_tokens
from .raptor_index import RaptorLiteIndex


client = OpenAI()



def read_txt_file(path: Path) -> str:

    return path.read_text(encoding="utf-8")


def read_pdf_file(path: Path) -> str:

    reader = PdfReader(str(path))
    pages_text: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n\n".join(pages_text)


def read_raw_texts(course_id: str) -> List[Dict[str, Any]]:
    course_dir = Path("data") / course_id
    raw_dir = course_dir / "raw"

    items: List[Dict[str, Any]] = []
    
    for path in raw_dir.glob("*.txt"):
        text = read_txt_file(path)
        items.append(
            {
                "source_id": path.name,
                "text": text,
            }
        )

    for path in raw_dir.glob("*.pdf"):
        text = read_pdf_file(path)
        items.append(
            {
                "source_id": path.name,
                "text": text,
            }
        )

    return items


def chunk_text(
    text: str,
    source_id: str,
    course_id: str,
    target_chunk_tokens: int = 300,
    start_chunk_index: int = 0,
) -> List[Dict[str, Any]]:
    model_for_chunking = settings.openai_model

    chunks_raw = split_into_token_chunks(
        text=text,
        max_tokens=target_chunk_tokens,
        model=model_for_chunking,
    )

    chunks: List[Dict[str, Any]] = []
    chunk_index = start_chunk_index

    for chunk_text_str in chunks_raw:
        num_tokens = count_tokens(chunk_text_str, model_for_chunking)

        chunks.append(
            {
                "id": f"chunk_{chunk_index:05d}",
                "course_id": course_id,
                "source_id": source_id,
                "position": {
                    "chunk_index": chunk_index,
                },
                "level": 0,
                "text": chunk_text_str,
                "num_tokens": num_tokens,
                "parent_ids": [],
            }
        )
        chunk_index += 1

    return chunks


def build_chunks_for_course(course_id: str) -> List[Dict[str, Any]]:

    source_items = read_raw_texts(course_id)
    all_chunks: List[Dict[str, Any]] = []

    next_chunk_index = 0  

    for item in source_items:
        source_id = item["source_id"]
        text = item["text"]

        chunks = chunk_text(
            text=text,
            source_id=source_id,
            course_id=course_id,
            target_chunk_tokens=300,
            start_chunk_index=next_chunk_index,
        )
        all_chunks.extend(chunks)
        next_chunk_index += len(chunks)

    return all_chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def summarize_texts(texts: List[str]) -> str:

    if not texts:
        return ""

    joined = "\n\n".join(texts[:8])

    prompt = (
        "You are helping to build a hierarchical index for a university course.\n"
        "Read the following fragments and write a short, concise summary (3-5 sentences)\n"
        "that describes the main topic and key concepts. Do not add extra information.\n\n"
        f"{joined}\n\n"
        "Summary:"
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()



def build_level_1_nodes(
    course_id: str,
    chunks: List[Dict[str, Any]],
    group_size: int = 5,
) -> List[Dict[str, Any]]:

    nodes: List[Dict[str, Any]] = []
    node_index = 0

    # разбиваем список чанков на группы
    for i in range(0, len(chunks), group_size):
        group = chunks[i : i + group_size]
        if not group:
            continue

        group_texts = [c["text"] for c in group]
        summary = summarize_texts(group_texts)

        node_id = f"node_l1_{node_index:04d}"
        children_ids = [c["id"] for c in group]

        node = {
            "id": node_id,
            "course_id": course_id,
            "level": 1,
            "summary": summary,
            "children_ids": children_ids,
            "embedding_index": node_index,
        }
        nodes.append(node)

        for c in group:
            parents = c.get("parent_ids", [])
            parents.append(node_id)
            c["parent_ids"] = parents

        node_index += 1

    return nodes


def build_raptor_lite_index(course_id: str) -> None:

    course_dir = Path("data") / course_id
    index_dir = course_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks_for_course(course_id)
    chunk_texts = [c["text"] for c in chunks]

    embeddings_level_0 = embed_texts(chunk_texts)

    nodes_level_1 = build_level_1_nodes(course_id, chunks, group_size=5)
    node_summaries = [n["summary"] for n in nodes_level_1]

    embeddings_level_1 = embed_texts(node_summaries)

    chunks_path = index_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    nodes_level_1_path = index_dir / "nodes_level_1.jsonl"
    with nodes_level_1_path.open("w", encoding="utf-8") as f:
        for n in nodes_level_1:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    emb0_path = index_dir / "embeddings_level_0.npy"
    emb1_path = index_dir / "embeddings_level_1.npy"
    np.save(emb0_path, embeddings_level_0)
    np.save(emb1_path, embeddings_level_1)

    meta = {
        "course_id": course_id,
        "index_version": 1,
        "levels": [0, 1],
        "files": {
            "chunks": chunks_path.name,
            "nodes": {
                "1": nodes_level_1_path.name,
            },
            "embeddings": {
                "0": emb0_path.name,
                "1": emb1_path.name,
            },
        },
        "embedding_model": settings.embedding_model,
        "summary_llm_model": settings.openai_model,
    }

    meta_path = index_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def print_index_stats(course_id: str) -> None:

    course_dir = Path("data") / course_id
    index = RaptorLiteIndex.from_disk(course_dir)

    num_chunks = len(index.chunks_by_id)
    num_nodes = len(index.nodes_by_id)

    print(f"Course: {course_id}")
    print(f"Chunks (level 0): {num_chunks}")
    print(f"Summary nodes (level 1): {num_nodes}")
    print(f"Embeddings level 0 shape: {index.embeddings_level_0.shape}")
    print(f"Embeddings level 1 shape: {index.embeddings_level_1.shape}")

    # покажем одно summary для sanity-check
    if index.nodes_by_id:
        first_node = next(iter(index.nodes_by_id.values()))
        print("\nExample level-1 summary node:")
        print(f"  id: {first_node.id}")
        print(f"  children: {len(first_node.children_ids)} chunks")
        print("  summary:")
        print("  ", first_node.summary[:500], "...")


if __name__ == "__main__":
    import sys
    course_id = sys.argv[1] if len(sys.argv) > 1 else "ir-2024"
    build_raptor_lite_index(course_id)
    print_index_stats(course_id)