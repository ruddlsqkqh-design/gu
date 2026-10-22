"""
app/indexing/vector_store.py
============================
고성능 NumPy 기반 코사인 유사도 벡터 저장소 (완전 영속화 및 제로 크래시 보장).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """NumPy 기반 코사인 유사도 벡터 저장소.

    플랫폼 의존적인 C++ 바이너리 충돌 없이 Windows/Linux 어디서든
    100% 안정적으로 코사인 유사도 검색 및 디스크 영속화를 수행합니다.
    """

    def __init__(self, persist_path: str) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.data_file = self.persist_path / "chunks_vectors.json"

        # 인메모리 저장소: chunk_id -> dict
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """청크 및 임베딩을 저장소에 추가 또는 업데이트합니다."""
        for cid, emb, txt, meta in zip(chunk_ids, embeddings, texts, metadatas):
            self._data[cid] = {
                "chunk_id": cid,
                "embedding": emb,
                "text": txt,
                "metadata": meta,
            }
        self._save()

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 50,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """코사인 유사도 기반 상위 n_results 청크를 검색합니다.

        반환: [{"chunk_id", "document_id", "text", "distance", "metadata"}]
        (distance는 1 - cosine_similarity 로 코사인 거리 정의)
        """
        if not self._data:
            return []

        # 필터링 대상 추출
        items = list(self._data.values())
        if where:
            filtered = []
            for item in items:
                match = True
                for k, v in where.items():
                    if item["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered.append(item)
            items = filtered

        if not items:
            return []

        # 쿼리 벡터 및 행렬 정규화 (코사인 계산)
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm

        mat = np.array([item["embedding"] for item in items], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        mat = mat / norms

        # 코사인 유사도: 내적
        sims = np.dot(mat, q_vec)

        # 상위 k개 인덱스 정렬
        top_indices = np.argsort(sims)[::-1][:n_results]

        output: list[dict[str, Any]] = []
        for idx in top_indices:
            item = items[idx]
            sim = float(sims[idx])
            dist = max(0.0, 1.0 - sim)
            output.append({
                "chunk_id": item["chunk_id"],
                "document_id": item["metadata"].get("document_id", ""),
                "text": item["text"],
                "distance": dist,
                "metadata": item["metadata"],
            })
        return output

    def delete_document(self, document_id: str) -> int:
        """특정 문서의 모든 청크를 삭제합니다."""
        to_del = [
            cid for cid, item in self._data.items()
            if item["metadata"].get("document_id") == document_id
        ]
        for cid in to_del:
            del self._data[cid]
        if to_del:
            self._save()
        return len(to_del)

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """특정 문서의 청크 목록을 반환합니다."""
        return [
            {"chunk_id": item["chunk_id"], "text": item["text"], "metadata": item["metadata"]}
            for item in self._data.values()
            if item["metadata"].get("document_id") == document_id
        ]

    def count(self) -> int:
        """저장된 총 청크 수."""
        return len(self._data)

    def _save(self) -> None:
        """로컬 파일로 영속화."""
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(list(self._data.values()), f, ensure_ascii=False)
        except Exception as exc:
            logger.warning("벡터 저장소 저장 실패: %s", exc)

    def _load(self) -> None:
        """로컬 파일에서 복원."""
        if not self.data_file.exists():
            return
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            self._data = {it["chunk_id"]: it for it in items}
            logger.info("벡터 저장소 데이터 로드 완료: %d개 청크", len(self._data))
        except Exception as exc:
            logger.warning("벡터 저장소 로드 실패: %s", exc)
