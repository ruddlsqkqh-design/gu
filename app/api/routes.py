"""
app/api/routes.py
=================
기술명세서 SPEC 2 §7 API 계약 구현.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Any
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.models.document import DocumentModel, DocumentStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# 요청 및 응답 모델
class AnalysisRequest(BaseModel):
    query_document_id: str = Field(..., description="분석 대상 문서 ID")
    corpus_scope: dict[str, Any] | None = Field(default=None, description="비교 대상 범위/필터")
    top_k: int = Field(default=10, description="반환할 상위 후보 수")
    policy_version: str = Field(default="similarity-policy-0.1", description="유사도 정책 버전")


class ReviewRequest(BaseModel):
    analysis_id: str
    candidate_id: str
    decision_label: str  # high_similarity | partial | low | hold
    rationale: str


# 서비스 주입 컨테이너
_services: dict[str, Any] = {}


def init_services(
    pipeline: Any,
    orchestrator: Any,
    vector_store: Any,
    bm25_index: Any,
    embedder: Any,
) -> None:
    _services.update({
        "pipeline": pipeline,
        "orchestrator": orchestrator,
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "embedder": embedder,
    })


@router.post("/v1/documents", summary="문서 등록 (OCR & 인덱싱)")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """PDF 문서를 업로드하여 OCR 파이프라인 및 백그라운드 색인을 수행합니다."""
    pipeline = _services.get("pipeline")
    if not pipeline:
        raise HTTPException(503, "OCR 파이프라인 서비스가 초기화되지 않았습니다.")

    content = await file.read()
    filename = file.filename or "unknown.pdf"

    try:
        doc = await pipeline.process(content, filename)
    except Exception as exc:
        raise HTTPException(500, f"OCR 처리 실패: {exc}")

    # 백그라운드 벡터 및 BM25 인덱싱
    background_tasks.add_task(_index_document, doc.document_id)

    return {
        "document_id": doc.document_id,
        "status": doc.status,
        "page_count": doc.page_count,
        "file_hash": doc.file_hash,
    }


@router.get("/v1/documents/{id}", summary="문서 상태/메타데이터 조회")
async def get_document(id: str):
    pipeline = _services.get("pipeline")
    if not pipeline:
        raise HTTPException(503, "서비스 준비 중")

    doc = pipeline.get_status(id)
    if not doc:
        raise HTTPException(404, "문서를 찾을 수 없습니다.")
    return doc.model_dump()


@router.post("/v1/analyses", summary="유사도 분석 요청")
async def create_analysis(req: AnalysisRequest, background_tasks: BackgroundTasks):
    orchestrator = _services.get("orchestrator")
    pipeline = _services.get("pipeline")

    if not orchestrator or not pipeline:
        raise HTTPException(503, "에이전트 오케스트레이터 서비스 준비 중")

    chunks = pipeline.get_chunks(req.query_document_id)
    if not chunks:
        raise HTTPException(404, "분석할 문서 청크를 찾을 수 없습니다.")

    chunk_dicts = [c.model_dump() for c in chunks]

    # 비동기로 멀티 에이전트 분석 수행
    analysis_id = f"analysis-{uuid.uuid4().hex[:8]}"

    async def _run():
        await orchestrator.analyze(
            query_document_id=req.query_document_id,
            query_chunks=chunk_dicts,
            corpus_scope=req.corpus_scope,
            top_k=req.top_k,
        )

    background_tasks.add_task(_run)

    return {
        "analysis_id": analysis_id,
        "query_document_id": req.query_document_id,
        "status": "QUEUED",
    }


@router.get("/v1/analyses/{id}", summary="분석 상태/요약 조회")
async def get_analysis(id: str):
    orchestrator = _services.get("orchestrator")
    if not orchestrator:
        raise HTTPException(503, "서비스 준비 중")

    res = orchestrator.get_result(id)
    if not res:
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다.")

    return {
        "analysis_id": res.analysis_id,
        "query_document_id": res.query_document_id,
        "status": res.status,
        "top_candidates_count": len(res.candidates),
        "created_at": res.created_at,
        "completed_at": res.completed_at,
    }


@router.get("/v1/analyses/{id}/report", summary="상세 근거 리포트")
async def get_analysis_report(id: str):
    orchestrator = _services.get("orchestrator")
    if not orchestrator:
        raise HTTPException(503, "서비스 준비 중")

    res = orchestrator.get_result(id)
    if not res:
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다.")

    return {
        "analysis_id": res.analysis_id,
        "query_document_id": res.query_document_id,
        "status": res.status,
        "candidates": res.candidates,
        "policy_version": res.policy_version,
        "embedding_model_version": res.embedding_model_version,
        "audit_trace": res.audit_trace,
    }


@router.post("/v1/reviews", summary="전문가 피드백 기록")
async def submit_review(req: ReviewRequest):
    review_path = Path("./storage/reviews.jsonl")
    review_path.parent.mkdir(parents=True, exist_ok=True)

    record = req.model_dump()
    with open(review_path, "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "SUCCESS", "message": "피드백이 기록되었습니다."}


async def _index_document(document_id: str) -> None:
    """백그라운드 임베딩 및 인덱싱."""
    pipeline = _services.get("pipeline")
    vector_store = _services.get("vector_store")
    bm25_index = _services.get("bm25_index")
    embedder = _services.get("embedder")

    if not all([pipeline, vector_store, bm25_index, embedder]):
        logger.warning("인덱싱 필수 서비스 누락으로 인덱싱 건너뜀")
        return

    chunks = pipeline.get_chunks(document_id)
    if not chunks:
        return

    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "document_id": c.document_id,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "chunk_type": c.chunk_type.value if hasattr(c.chunk_type, "value") else str(c.chunk_type),
            "section_path": " > ".join(c.section_path) if c.section_path else "",
        }
        for c in chunks
    ]

    try:
        embeddings = await embedder.embed_texts(texts)
        vector_store.upsert_chunks(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
        )
        bm25_index.add_chunks(
            chunk_ids=chunk_ids,
            texts=texts,
            doc_ids=[document_id] * len(chunk_ids),
        )
        logger.info("문서 %s 인덱싱 완료 (%d개 청크)", document_id, len(chunks))
    except Exception as exc:
        logger.error("인덱싱 실패 %s: %s", document_id, exc)
