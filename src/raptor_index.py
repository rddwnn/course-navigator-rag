from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from pathlib import Path
import json
import numpy as np 

@dataclass
class Chunk:
    id: str
    course_id: str
    source_id: str
    position: Dict[str, Any]
    level: int
    text: str
    parent_ids: List[str] 

@dataclass
class SummaryNode:
    id: str
    course_id: str
    level: int
    summary: str
    children_ids: List[str]
    embedding_index: int

@dataclass
class CourseIndexMeta:
    course_id: str
    levels: List[int]
    chunks_file: str
    nodes_level_1_file: str
    embeddings_level_0_file: str
    embeddings_level_1_file: str
    embedding_model: str
    summary_llm_model: str

@dataclass
class RaptorLiteIndex:
    def __init__(
        self,
        base_dir: Path,
        meta: CourseIndexMeta,
        chunks_by_id: Dict[str, Chunk],
        nodes_by_id: Dict[str, SummaryNode],
        embeddings_level_0: np.ndarray,
        embeddings_level_1: np.ndarray,
    ) -> None:
        self.base_dir = base_dir
        self.meta = meta
        self.chunks_by_id = chunks_by_id
        self.nodes_by_id = nodes_by_id
        self.embeddings_level_0 = embeddings_level_0
        self.embeddings_level_1 = embeddings_level_1

    @classmethod
    def from_disk(cls, course_dir: Path) -> "RaptorLiteIndex":
        index_dir = course_dir / "index"
        meta_path = index_dir / "meta.json"

        with meta_path.open("r", encoding="utf-8") as f:
            raw_meta = json.load(f)

        meta = CourseIndexMeta(
            course_id=raw_meta["course_id"],
            levels=raw_meta["levels"],
            chunks_file=raw_meta["files"]["chunks"],
            nodes_level_1_file=raw_meta["files"]["nodes"]["1"],
            embeddings_level_0_file=raw_meta["files"]["embeddings"]["0"],
            embeddings_level_1_file=raw_meta["files"]["embeddings"]["1"],
            embedding_model=raw_meta["embedding_model"],
            summary_llm_model=raw_meta["summary_llm_model"],
        )

        chunks_by_id = _load_chunks(index_dir / meta.chunks_file)
        nodes_by_id = _load_nodes_level_1(index_dir / meta.nodes_level_1_file)
        embeddings_level_0 = np.load(index_dir / meta.embeddings_level_0_file)
        embeddings_level_1 = np.load(index_dir / meta.embeddings_level_1_file)

        return cls(
            base_dir=course_dir,
            meta=meta,
            chunks_by_id=chunks_by_id,
            nodes_by_id=nodes_by_id,
            embeddings_level_0=embeddings_level_0,
            embeddings_level_1=embeddings_level_1,
        )


def _load_chunks(path: Path) -> Dict[str, Chunk]:
    chunks_by_id: Dict[str, Chunk] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)
            chunk = Chunk(
                id=raw["id"],
                course_id=raw["course_id"],
                source_id=raw["source_id"],
                position=raw.get("position", {}),
                level=raw.get("level", 0),
                text=raw["text"],
                parent_ids=raw.get("parent_ids", []),
            )
            chunks_by_id[chunk.id] = chunk
    return chunks_by_id


def _load_nodes_level_1(path: Path) -> Dict[str, SummaryNode]:
  
    nodes_by_id: Dict[str, SummaryNode] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)
            node = SummaryNode(
                id=raw["id"],
                course_id=raw["course_id"],
                level=raw["level"],
                summary=raw["summary"],
                children_ids=raw.get("children_ids", []),
                embedding_index=raw["embedding_index"],
            )
            nodes_by_id[node.id] = node
    return nodes_by_id
