"""
E2E 파이프라인 통합 테스트 (실제 생성된 PDF 파싱 -> 청킹 -> 검색 -> 에이전트 분석)
"""

import asyncio
from pathlib import Path
from app.config import get_settings
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.agents.orchestrator import AgentOrchestrator
from app.ocr.pipeline import OCRPipeline

async def run_e2e():
    settings = get_settings()
    pipeline = OCRPipeline(settings)
    vstore = VectorStore(settings.CHROMA_DB_PATH)
    bm25 = BM25Index("./storage/bm25")
    embedder = EmbeddingService(
        settings.GOOGLE_CLOUD_PROJECT,
        settings.GOOGLE_CLOUD_LOCATION,
        settings.VERTEX_AI_EMBEDDING_MODEL,
    )
    orch = AgentOrchestrator(settings, vstore, bm25, embedder, None)

    # 1. 문서 A 처리
    path_a = Path("./storage/sample_docs/plan_A_ocr_engine.pdf")
    with open(path_a, "rb") as f:
        bytes_a = f.read()
    print("Processing Document A...")
    doc_a = await pipeline.process(bytes_a, "plan_A_ocr_engine.pdf", tenant_id="inst_kpc")
    chunks_a = pipeline.get_chunks(doc_a.document_id)
    print(f"Doc A processed: {doc_a.document_id}, page_count: {doc_a.page_count}, chunks: {len(chunks_a)}")

    # 인덱싱
    texts_a = [c.text for c in chunks_a]
    ids_a = [c.chunk_id for c in chunks_a]
    embs_a = await embedder.embed_texts(texts_a)
    vstore.upsert_chunks(ids_a, embs_a, texts_a, [{"document_id": doc_a.document_id} for _ in chunks_a])
    bm25.add_chunks(ids_a, texts_a, [doc_a.document_id]*len(chunks_a))

    # 2. 문서 B 처리
    path_b = Path("./storage/sample_docs/plan_B_similarity_agent.pdf")
    with open(path_b, "rb") as f:
        bytes_b = f.read()
    print("Processing Document B...")
    doc_b = await pipeline.process(bytes_b, "plan_B_similarity_agent.pdf", tenant_id="inst_kpc")
    chunks_b = pipeline.get_chunks(doc_b.document_id)
    print(f"Doc B processed: {doc_b.document_id}, page_count: {doc_b.page_count}, chunks: {len(chunks_b)}")

    # 3. 문서 B를 쿼리로 문서 A와의 유사도 판별 분석 실행
    print("Running Multi-Agent Similarity Analysis on Document B...")
    query_chunks = [c.model_dump() for c in chunks_b]
    analysis_result = await orch.analyze(
        query_document_id=doc_b.document_id,
        query_chunks=query_chunks,
        top_k=3,
    )
    print("\n================ 분석 결과 리포트 ================")
    print(f"Analysis ID: {analysis_result.analysis_id}")
    print(f"Status: {analysis_result.status}")
    print(f"Found Candidates: {len(analysis_result.candidates)}")
    for cand in analysis_result.candidates:
        print(f"\n- Candidate Document ID: {cand['document_id']}")
        print(f"  Overall Score: {cand['overall_similarity_score']} / 100")
        print(f"  Dimension Scores: {cand['dimension_scores']}")
        print(f"  Review Recommendation: {cand['review_recommendation']}")
        print(f"  Differences: {cand['differences']}")

if __name__ == "__main__":
    asyncio.run(run_e2e())
