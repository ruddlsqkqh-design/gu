"""
app/ocr/adapters/document_ai.py
================================
Google Cloud Document AI OCR 어댑터.

스캔 PDF 또는 이미지에서 텍스트, 표, 신뢰도를 추출합니다.
API 설정이 없을 경우 NotConfiguredError를 발생시켜 graceful degradation을
지원합니다. tenacity를 이용한 지수 백오프 재시도 로직을 포함합니다.

엔진명: google-document-ai
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.ocr.adapters.base import (
    BaseOCRAdapter,
    NotConfiguredError,
    OCRProcessingError,
    PageOCRResult,
)

logger = logging.getLogger(__name__)

_ENGINE_NAME = "google-document-ai"
_MAX_RETRY_ATTEMPTS = 3
_RETRY_WAIT_MIN = 2
_RETRY_WAIT_MAX = 10

# 스레드 풀 (동기 SDK 호출을 비동기로 래핑)
_executor = ThreadPoolExecutor(max_workers=4)


class DocumentAIAdapter(BaseOCRAdapter):
    """Google Cloud Document AI OCR 어댑터.

    비동기 방식으로 Document AI API를 호출하며, 동기 SDK는 ThreadPoolExecutor를
    통해 이벤트 루프를 블로킹하지 않도록 처리합니다.

    Args:
        project_id: GCP 프로젝트 ID
        location: Document AI 프로세서 리전 (예: "us")
        processor_id: Document AI 프로세서 ID
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        processor_id: str,
    ) -> None:
        self._project_id = project_id
        self._location = location
        self._processor_id = processor_id
        self._client: Any | None = None
        self._configured = bool(project_id and location and processor_id)
        self._model_version_str = "pretrained-v1.3"

        if self._configured:
            self._init_client()

    def _init_client(self) -> None:
        """Document AI 클라이언트를 초기화합니다."""
        try:
            from google.cloud import documentai  # type: ignore

            client_options = {
                "api_endpoint": f"{self._location}-documentai.googleapis.com"
            }
            self._client = documentai.DocumentProcessorServiceClient(
                client_options=client_options
            )
            logger.info("[%s] Document AI 클라이언트 초기화 완료", _ENGINE_NAME)
        except ImportError as exc:
            logger.warning(
                "[%s] google-cloud-documentai 패키지 미설치: %s", _ENGINE_NAME, exc
            )
            self._configured = False
        except Exception as exc:
            logger.warning("[%s] 클라이언트 초기화 실패: %s", _ENGINE_NAME, exc)
            self._configured = False

    @property
    def engine_name(self) -> str:
        """OCR 엔진 식별자."""
        return _ENGINE_NAME

    @property
    def model_version(self) -> str:
        """Document AI 모델 버전."""
        return self._model_version_str

    def _check_configured(self) -> None:
        """설정 완료 여부를 확인합니다.

        Raises:
            NotConfiguredError: 필수 설정값이 없을 경우
        """
        if not self._configured or self._client is None:
            raise NotConfiguredError(
                _ENGINE_NAME,
                "DOCUMENT_AI_PROCESSOR_ID / GOOGLE_CLOUD_PROJECT / "
                "DOCUMENT_AI_PROCESSOR_LOCATION",
            )

    async def process_page(
        self, page_image_bytes: bytes, page_number: int
    ) -> PageOCRResult:
        """단일 페이지 이미지를 Document AI로 처리합니다.

        Args:
            page_image_bytes: 페이지 이미지 바이트 (PNG/JPEG)
            page_number: 페이지 번호 (1-based)

        Returns:
            PageOCRResult: OCR 처리 결과

        Raises:
            NotConfiguredError: API 설정 누락 시
            OCRProcessingError: 처리 실패 시
        """
        self._check_configured()
        try:
            results = await self._call_document_ai_async(
                page_image_bytes, mime_type="image/png"
            )
            if not results:
                return PageOCRResult(
                    page_number=page_number,
                    text="",
                    blocks=[],
                    tables=[],
                    confidence=0.0,
                    quality_flags=["empty_response"],
                )
            result = results[0]
            result.page_number = page_number
            return result
        except NotConfiguredError:
            raise
        except Exception as exc:
            raise OCRProcessingError(
                _ENGINE_NAME, str(exc), page=page_number
            ) from exc

    async def process_pdf(
        self, pdf_bytes: bytes, doc_id: str
    ) -> list[PageOCRResult]:
        """PDF 전체를 Document AI로 처리합니다.

        Args:
            pdf_bytes: PDF 파일 바이트
            doc_id: 문서 식별자 (로깅용)

        Returns:
            list[PageOCRResult]: 페이지별 OCR 결과 목록

        Raises:
            NotConfiguredError: API 설정 누락 시
            OCRProcessingError: 처리 실패 시
        """
        self._check_configured()
        logger.info("[%s] 문서 %s: Document AI 처리 시작", _ENGINE_NAME, doc_id)

        try:
            results = await self._call_document_ai_async(
                pdf_bytes, mime_type="application/pdf"
            )
            logger.info(
                "[%s] 문서 %s: %d 페이지 처리 완료",
                _ENGINE_NAME,
                doc_id,
                len(results),
            )
            return results
        except NotConfiguredError:
            raise
        except Exception as exc:
            raise OCRProcessingError(
                _ENGINE_NAME, f"PDF 처리 실패 (doc_id={doc_id}): {exc}"
            ) from exc

    async def _call_document_ai_async(
        self, content: bytes, mime_type: str
    ) -> list[PageOCRResult]:
        """Document AI API를 ThreadPoolExecutor에서 비동기로 호출합니다.

        Args:
            content: 처리할 파일 바이트
            mime_type: 파일 MIME 타입

        Returns:
            list[PageOCRResult]: 페이지별 결과
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._call_document_ai_sync, content, mime_type
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=_RETRY_WAIT_MIN, max=_RETRY_WAIT_MAX),
        reraise=True,
    )
    def _call_document_ai_sync(
        self, content: bytes, mime_type: str
    ) -> list[PageOCRResult]:
        """Document AI API 동기 호출 (재시도 데코레이터 적용).

        Args:
            content: 처리할 파일 바이트
            mime_type: 파일 MIME 타입

        Returns:
            list[PageOCRResult]: 페이지별 결과
        """
        from google.cloud import documentai  # type: ignore

        processor_name = self._client.processor_path(
            self._project_id, self._location, self._processor_id
        )

        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=processor_name, raw_document=raw_document
        )

        response = self._client.process_document(request=request)
        document = response.document

        # 모델 버전 업데이트
        if hasattr(response, "human_review_status"):
            pass  # 향후 확장용

        return self._convert_document(document)

    def _convert_document(self, document: Any) -> list[PageOCRResult]:
        """Document AI 응답을 PageOCRResult 목록으로 변환합니다.

        Args:
            document: Document AI Document 객체

        Returns:
            list[PageOCRResult]: 변환된 페이지 결과 목록
        """
        from google.cloud import documentai  # type: ignore

        pages: list[PageOCRResult] = []
        full_text = document.text or ""

        for page in document.pages:
            page_num = page.page_number  # 1-based
            page_width = float(page.dimension.width) if page.dimension.width else 1.0
            page_height = float(page.dimension.height) if page.dimension.height else 1.0

            # ── 텍스트 블록 추출 ───────────────────────────────────────────
            blocks: list[dict[str, Any]] = []
            confidences: list[float] = []

            for block in page.blocks:
                block_text = self._extract_text_from_layout(
                    block.layout, full_text
                )
                if not block_text.strip():
                    continue

                conf = float(block.layout.confidence) if block.layout.confidence else 0.5
                confidences.append(conf)
                bbox = self._convert_bbox(block.layout, page_width, page_height)

                blocks.append(
                    {
                        "text": block_text.strip(),
                        "bbox": bbox,
                        "confidence": conf,
                        "block_type": "paragraph",
                    }
                )

            # ── 표 추출 ────────────────────────────────────────────────────
            tables: list[dict[str, Any]] = []
            for tbl_idx, table in enumerate(page.tables):
                cells: list[list[dict[str, Any]]] = []

                all_rows = list(table.header_rows) + list(table.body_rows)
                for row_idx, row in enumerate(all_rows):
                    cell_row: list[dict[str, Any]] = []
                    for col_idx, cell in enumerate(row.cells):
                        cell_text = self._extract_text_from_layout(
                            cell.layout, full_text
                        )
                        cell_bbox = self._convert_bbox(
                            cell.layout, page_width, page_height
                        )
                        cell_row.append(
                            {
                                "text": cell_text.strip(),
                                "row": row_idx,
                                "col": col_idx,
                                "bbox": cell_bbox,
                            }
                        )
                    cells.append(cell_row)

                tbl_bbox = self._convert_bbox(table.layout, page_width, page_height)
                tables.append(
                    {
                        "cells": cells,
                        "bbox": tbl_bbox,
                        "page": page_num,
                        "table_index": tbl_idx,
                    }
                )

            # ── 페이지 텍스트 및 신뢰도 집계 ──────────────────────────────
            page_text = self._build_page_text(blocks, tables)
            avg_conf = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            pages.append(
                PageOCRResult(
                    page_number=page_num,
                    text=page_text,
                    blocks=blocks,
                    tables=tables,
                    confidence=avg_conf,
                    raw_response=None,
                )
            )

        return pages

    @staticmethod
    def _extract_text_from_layout(layout: Any, full_text: str) -> str:
        """Document AI Layout의 text_anchor를 사용해 텍스트를 추출합니다.

        Args:
            layout: Document AI Layout 객체
            full_text: 문서 전체 텍스트

        Returns:
            str: 추출된 텍스트
        """
        if not layout or not layout.text_anchor:
            return ""
        segments = layout.text_anchor.text_segments
        text_parts: list[str] = []
        for segment in segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index) if segment.end_index else 0
            text_parts.append(full_text[start:end])
        return "".join(text_parts)

    @staticmethod
    def _convert_bbox(
        layout: Any, page_width: float, page_height: float
    ) -> list[float]:
        """Document AI bounding_poly를 정규화된 bbox로 변환합니다.

        Args:
            layout: Document AI Layout 객체
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            list[float]: [x0, y0, x1, y1] 정규화 좌표
        """
        if not layout or not layout.bounding_poly:
            return [0.0, 0.0, 1.0, 1.0]

        vertices = layout.bounding_poly.normalized_vertices
        if not vertices:
            vertices_abs = layout.bounding_poly.vertices
            if not vertices_abs:
                return [0.0, 0.0, 1.0, 1.0]
            xs = [v.x / page_width for v in vertices_abs]
            ys = [v.y / page_height for v in vertices_abs]
        else:
            xs = [v.x for v in vertices]
            ys = [v.y for v in vertices]

        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _build_page_text(
        blocks: list[dict], tables: list[dict]
    ) -> str:
        """블록과 표 텍스트를 결합하여 페이지 전체 텍스트를 생성합니다.

        Args:
            blocks: 텍스트 블록 목록
            tables: 표 목록

        Returns:
            str: 결합된 페이지 텍스트
        """
        items: list[dict[str, Any]] = []
        for block in blocks:
            items.append({"y0": block["bbox"][1], "text": block["text"]})
        for table in tables:
            table_lines: list[str] = []
            for row in table["cells"]:
                table_lines.append("\t".join(cell["text"] for cell in row))
            items.append({"y0": table["bbox"][1], "text": "\n".join(table_lines)})

        items.sort(key=lambda x: x["y0"])
        return "\n".join(item["text"] for item in items).strip()
