"""
app/ocr/pipeline.py
===================
OCR 파이프라인 오케스트레이터.

전체 처리 단계:
1. 수집 & 파일 검증 (해시 계산, 중복/멱등성 검사)
2. 원본 보관 (storage/originals)
3. 사전 분석 (텍스트 레이어 유무에 따른 어댑터 자동 선택)
4. OCR 및 레이아웃 분석 실행
5. 문서 정규화 및 섹션 트리 구축
6. 구조화 청킹 생성 (SPEC 1 §5 정책)
7. 품질 검사 (빈 페이지, 낮은 신뢰도 등)
8. 결과 영속화 (storage/ocr_results)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.models.chunk import ChunkModel
from app.models.document import DocumentModel, DocumentStatus
from app.ocr.adapters.base import BaseOCRAdapter, PageOCRResult
from app.ocr.adapters.pdf_parser import PDFParserAdapter
from app.ocr.chunker import ChunkingEngine
from app.ocr.layout import LayoutAnalyzer
from app.ocr.normalizer import normalize_text
from app.ocr.quality import QualityChecker

logger = logging.getLogger(__name__)


class OCRPipeline:
    """기술사업계획서 OCR 및 청킹 파이프라인."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        # 스토리지 디렉토리 준비
        self.originals_dir = Path(self.settings.ORIGINALS_PATH)
        self.results_dir = Path(self.settings.OCR_RESULTS_PATH)
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 컴포넌트 초기화
        self.layout_analyzer = LayoutAnalyzer()
        self.chunker = ChunkingEngine(
            min_tokens=self.settings.CHUNK_MIN_TOKENS,
            max_tokens=self.settings.CHUNK_MAX_TOKENS,
            overlap_tokens=self.settings.CHUNK_OVERLAP_TOKENS,
        )
        self.quality_checker = QualityChecker()

        # 기본 PDF 파서
        self.default_adapter = PDFParserAdapter()

        # Document AI / Vision 어댑터 로드 시도
        self.doc_ai_adapter = None
        self.vision_adapter = None
        self._init_cloud_adapters()

    def _init_cloud_adapters(self) -> None:
        """클라우드 어댑터 로드 시도 (미설정 또는 라이브러리 부재 시 graceful fallback)."""
        try:
            from app.ocr.adapters.document_ai import DocumentAIAdapter
            self.doc_ai_adapter = DocumentAIAdapter(
                project_id=self.settings.GOOGLE_CLOUD_PROJECT,
                location=self.settings.DOCUMENT_AI_PROCESSOR_LOCATION,
                processor_id=self.settings.DOCUMENT_AI_PROCESSOR_ID,
            )
        except Exception as exc:
            logger.debug("Document AI 어댑터 초기화 생략: %s", exc)

        try:
            from app.ocr.adapters.vision_ocr import VisionOCRAdapter
            self.vision_adapter = VisionOCRAdapter()
        except Exception as exc:
            logger.debug("Vision OCR 어댑터 초기화 생략: %s", exc)

    def _select_adapter(self, pdf_bytes: bytes) -> BaseOCRAdapter:
        """PDF 특성을 분석하여 최적의 OCR 어댑터를 선택합니다."""
        has_text = self.default_adapter.has_text_layer(pdf_bytes)
        if has_text:
            logger.info("텍스트 레이어 감지됨 -> pdfplumber 직접 파싱 사용")
            return self.default_adapter

        # 스캔 이미지 PDF인 경우
        if self.doc_ai_adapter and getattr(self.doc_ai_adapter, "is_available", False):
            logger.info("스캔 문서 감지됨 -> Document AI 어댑터 사용")
            return self.doc_ai_adapter
        elif self.vision_adapter and getattr(self.vision_adapter, "is_available", False):
            logger.info("스캔 문서 감지됨 -> Vision OCR 어댑터 사용")
            return self.vision_adapter

        logger.warning("클라우드 OCR 미구성 또는 사용 불가 -> 기본 파서로 폴백")
        return self.default_adapter

    async def process(
        self,
        file_bytes: bytes,
        filename: str,
        tenant_id: str = "default",
        force_reprocess: bool = False,
    ) -> DocumentModel:
        """문서 파일을 수집하고 OCR, 구조화, 청킹 파이프라인을 실행합니다."""
        # 1. 해시 계산 및 검증
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 멱등성 검사: 이미 처리된 동일 해시 문서가 있는지 확인
        if not force_reprocess:
            existing_doc = self._find_by_hash(file_hash)
            if existing_doc and existing_doc.status == DocumentStatus.SUCCEEDED:
                logger.info("이미 동일 해시로 처리된 문서 존재: %s", existing_doc.document_id)
                return existing_doc

        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        file_path = self.originals_dir / f"{doc_id}_{filename}"

        # 2. 원본 보관
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        doc = DocumentModel(
            document_id=doc_id,
            tenant_id=tenant_id,
            source_uri=str(file_path),
            original_filename=filename,
            file_hash=file_hash,
            status=DocumentStatus.RUNNING,
        )
        self._save_doc_meta(doc)

        try:
            # 3. 어댑터 선택 및 OCR 실행
            adapter = self._select_adapter(file_bytes)
            doc.mark_running(ocr_engine=adapter.engine_name, model_version=adapter.model_version)

            ocr_results = await adapter.process_pdf(file_bytes, doc_id=doc_id)
            page_count = len(ocr_results)

            # 4. 레이아웃 & 섹션 분석
            all_blocks = []
            for page in ocr_results:
                all_blocks.extend(page.blocks)

            section_tree = self.layout_analyzer.build_section_tree(all_blocks)
            section_map = self.layout_analyzer.build_block_section_map(all_blocks, section_tree)

            # 5. 청킹 생성
            chunks = self.chunker.chunk_document(
                document_id=doc_id,
                ocr_results=ocr_results,
                section_map=section_map,
            )

            # 6. 품질 검사
            quality_summary = self.quality_checker.check_document(ocr_results, chunks)

            # 7. 결과 저장
            self._save_results(doc_id, ocr_results, chunks)

            # 문서 완료 처리
            if quality_summary.needs_review:
                doc.mark_partial(page_count=page_count, quality_summary=quality_summary.to_dict())
            else:
                doc.mark_succeeded(page_count=page_count, quality_summary=quality_summary.to_dict())

            self._save_doc_meta(doc)
            return doc

        except Exception as exc:
            logger.error("파이프라인 실행 중 오류 발생: %s", exc, exc_info=True)
            doc.mark_failed(str(exc))
            self._save_doc_meta(doc)
            raise

    def get_status(self, document_id: str) -> DocumentModel | None:
        """문서 상태 및 메타데이터를 조회합니다."""
        meta_path = self.results_dir / f"{document_id}_meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentModel.model_validate(data)

    def get_chunks(self, document_id: str) -> list[ChunkModel]:
        """문서 청크 목록을 조회합니다."""
        chunks_path = self.results_dir / f"{document_id}_chunks.json"
        if not chunks_path.exists():
            return []
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChunkModel.model_validate(item) for item in data]

    def list_documents(self) -> list[DocumentModel]:
        """저장된 모든 문서 목록을 반환합니다."""
        docs = []
        for file in self.results_dir.glob("*_meta.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    docs.append(DocumentModel.model_validate(json.load(f)))
            except Exception:
                continue
        docs.sort(key=lambda d: d.created_at, reverse=True)
        return docs

    def _save_doc_meta(self, doc: DocumentModel) -> None:
        path = self.results_dir / f"{doc.document_id}_meta.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

    def _save_results(
        self,
        document_id: str,
        ocr_results: list[PageOCRResult],
        chunks: list[ChunkModel],
    ) -> None:
        # OCR 결과 저장
        ocr_path = self.results_dir / f"{document_id}_ocr.json"
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump([vars(r) for r in ocr_results], f, ensure_ascii=False, indent=2)

        # 청크 저장
        chunks_path = self.results_dir / f"{document_id}_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)

    def _find_by_hash(self, file_hash: str) -> DocumentModel | None:
        for doc in self.list_documents():
            if doc.file_hash == file_hash:
                return doc
        return None
