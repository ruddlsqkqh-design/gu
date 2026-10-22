"""
tests/test_pipeline.py
======================
KPC 솔루션 종합 단위 및 파이프라인 통합 테스트.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.agents.orchestrator import AgentOrchestrator
from app.ocr.chunker import ChunkingEngine
from app.ocr.layout import LayoutAnalyzer
from app.ocr.normalizer import normalize_text
from app.ocr.quality import QualityChecker
from app.ocr.adapters.base import PageOCRResult


class TestKPCPipeline(unittest.TestCase):
    def setUp(self):
        self.settings = get_settings()

    def test_01_normalizer(self):
        raw = "이것은　전각공백과   과도한   공백,\n\n\n줄바꿈\n테스트- \n입니다."
        norm = normalize_text(raw)
        self.assertNotIn("　", norm)
        self.assertNotIn("\n\n\n", norm)
        self.assertTrue(len(norm) > 0)

    def test_02_layout_analyzer(self):
        analyzer = LayoutAnalyzer()
        blocks = [
            {"text": "제 1 장 기술 개요", "bbox": [0, 0, 1, 0.1], "confidence": 1.0, "block_type": "paragraph"},
            {"text": "본 기술은 OCR 및 멀티 에이전트 유사도 판별 시스템입니다.", "bbox": [0, 0.1, 1, 0.2], "confidence": 1.0, "block_type": "paragraph"},
            {"text": "1.1 핵심 기술 원리", "bbox": [0, 0.2, 1, 0.3], "confidence": 1.0, "block_type": "paragraph"},
            {"text": "하이브리드 벡터 검색을 수행합니다.", "bbox": [0, 0.3, 1, 0.4], "confidence": 1.0, "block_type": "paragraph"},
        ]
        tree = analyzer.build_section_tree(blocks)
        self.assertTrue(len(tree) >= 1)
        section_map = analyzer.build_block_section_map(blocks, tree)
        self.assertIn(1, section_map)

    def test_03_chunker(self):
        chunker = ChunkingEngine(min_tokens=20, max_tokens=100, overlap_tokens=10)
        pages = [
            PageOCRResult(
                page_number=1,
                text="1. 기술 개발 목표\n본 과제는 차세대 OCR 및 멀티 에이전트 기반 판별 솔루션을 구축하는 것입니다.",
                blocks=[
                    {"text": "1. 기술 개발 목표", "bbox": [0.1, 0.1, 0.9, 0.15], "confidence": 1.0, "block_type": "heading1"},
                    {"text": "본 과제는 차세대 OCR 및 멀티 에이전트 기반 판별 솔루션을 구축하는 것입니다.", "bbox": [0.1, 0.2, 0.9, 0.3], "confidence": 1.0, "block_type": "paragraph"},
                ],
                tables=[],
                confidence=1.0,
            )
        ]
        chunks = chunker.chunk_document(document_id="doc-test-1", ocr_results=pages)
        self.assertTrue(len(chunks) >= 1)
        self.assertEqual(chunks[0].document_id, "doc-test-1")

    def test_04_quality_checker(self):
        checker = QualityChecker()
        empty_page = PageOCRResult(
            page_number=1,
            text="",
            blocks=[],
            tables=[],
            confidence=0.0,
        )
        flags = checker.check_page(empty_page)
        self.assertIn("empty_page", flags)

    def test_05_orchestrator_flow(self):
        async def _run():
            embedder = EmbeddingService(
                project=self.settings.GOOGLE_CLOUD_PROJECT,
                location=self.settings.GOOGLE_CLOUD_LOCATION,
                model_name=self.settings.VERTEX_AI_EMBEDDING_MODEL,
            )
            vstore = VectorStore(persist_path="./storage/test_chroma")
            bm25 = BM25Index(index_path="./storage/test_bm25")

            # 샘플 문서 등록
            doc_id = "doc-candidate-001"
            texts = [
                "인공지능 기반의 기술사업계획서 OCR 및 문서 청킹 기술입니다.",
                "딥러닝 모델과 하이브리드 벡터 검색을 활용하여 사업계획서 유사도를 측정합니다.",
            ]
            c_ids = ["chunk-1", "chunk-2"]
            embs = await embedder.embed_texts(texts)
            vstore.upsert_chunks(
                c_ids, embs, texts,
                [{"document_id": doc_id, "page_start": 1, "section_path": "개요"} for _ in texts]
            )
            bm25.add_chunks(c_ids, texts, [doc_id, doc_id])

            orchestrator = AgentOrchestrator(
                settings=self.settings,
                vector_store=vstore,
                bm25_index=bm25,
                embedder=embedder,
                llm_client=None,
            )

            query_chunks = [
                {
                    "chunk_id": "query-chk-1",
                    "document_id": "doc-new-001",
                    "text": "인공지능 OCR 엔진을 이용한 사업계획서 유사도 판별 연구",
                    "section_path": ["기술개요"],
                    "page_start": 1,
                }
            ]

            result = await orchestrator.analyze(
                query_document_id="doc-new-001",
                query_chunks=query_chunks,
                top_k=3,
            )
            self.assertEqual(result.status, "COMPLETED")
            self.assertTrue(len(result.candidates) >= 1)
            self.assertIn("overall_similarity_score", result.candidates[0])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
