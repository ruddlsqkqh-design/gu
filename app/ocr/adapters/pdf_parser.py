"""
app/ocr/adapters/pdf_parser.py
==============================
pdfplumber 기반 텍스트 PDF 직접 파싱 어댑터.

스캔 없이 텍스트 레이어가 포함된 PDF에서 텍스트, 표, 레이아웃 정보를
직접 추출합니다. 신뢰도는 1.0으로 고정됩니다.

엔진명: pdfplumber
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pdfplumber
from pdfplumber.page import Page

from app.ocr.adapters.base import (
    BaseOCRAdapter,
    OCRProcessingError,
    PageOCRResult,
)

logger = logging.getLogger(__name__)

_ENGINE_NAME = "pdfplumber"
_MODEL_VERSION = "0.11.x"

# 텍스트 레이어 판단 기준: 페이지당 최소 문자 수
_MIN_CHARS_FOR_TEXT_LAYER = 20


class PDFParserAdapter(BaseOCRAdapter):
    """pdfplumber를 사용한 텍스트 PDF 파싱 어댑터.

    텍스트 레이어가 있는 PDF를 직접 파싱하여 구조화된 OCR 결과를 반환합니다.
    이미지 기반 스캔 PDF에는 적합하지 않으며, 해당 경우에는
    DocumentAIAdapter 또는 VisionOCRAdapter를 사용해야 합니다.
    """

    @property
    def engine_name(self) -> str:
        """OCR 엔진 식별자."""
        return _ENGINE_NAME

    @property
    def model_version(self) -> str:
        """어댑터 버전."""
        return _MODEL_VERSION

    # ── 공개 메서드 ─────────────────────────────────────────────────────────

    async def process_page(
        self, page_image_bytes: bytes, page_number: int
    ) -> PageOCRResult:
        """단일 페이지 이미지 처리 (pdfplumber는 이미지 방식 미지원).

        pdfplumber는 PDF 스트림을 직접 파싱하므로, 이미지 단독 처리는
        지원하지 않습니다. PDF 전체 처리를 위해 process_pdf를 사용하세요.

        Raises:
            OCRProcessingError: 항상 발생 (이미지 처리 미지원)
        """
        raise OCRProcessingError(
            _ENGINE_NAME,
            "pdfplumber 어댑터는 이미지 단독 처리를 지원하지 않습니다. "
            "process_pdf()를 사용하세요.",
            page=page_number,
        )

    async def process_pdf(
        self, pdf_bytes: bytes, doc_id: str
    ) -> list[PageOCRResult]:
        """PDF 바이트를 파싱하여 페이지별 OCR 결과를 반환합니다.

        Args:
            pdf_bytes: PDF 파일 전체 바이트
            doc_id: 문서 식별자 (로깅용)

        Returns:
            list[PageOCRResult]: 페이지 순서로 정렬된 OCR 결과 목록

        Raises:
            OCRProcessingError: PDF 파싱 실패 시
        """
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            results: list[PageOCRResult] = []

            with pdfplumber.open(pdf_stream) as pdf:
                logger.info(
                    "[%s] 문서 %s: 총 %d 페이지 파싱 시작",
                    _ENGINE_NAME,
                    doc_id,
                    len(pdf.pages),
                )
                for page in pdf.pages:
                    try:
                        result = self._parse_page(page)
                        results.append(result)
                    except Exception as exc:
                        logger.warning(
                            "[%s] 문서 %s, 페이지 %d 파싱 오류: %s",
                            _ENGINE_NAME,
                            doc_id,
                            page.page_number,
                            exc,
                        )
                        # 빈 결과로 대체하여 나머지 페이지 처리 유지
                        results.append(
                            PageOCRResult(
                                page_number=page.page_number,
                                text="",
                                blocks=[],
                                tables=[],
                                confidence=0.0,
                                quality_flags=["parse_error"],
                            )
                        )

            logger.info(
                "[%s] 문서 %s: %d 페이지 파싱 완료", _ENGINE_NAME, doc_id, len(results)
            )
            return results

        except Exception as exc:
            raise OCRProcessingError(
                _ENGINE_NAME, f"PDF 파싱 실패: {exc}"
            ) from exc

    def has_text_layer(self, pdf_bytes: bytes, sample_pages: int = 3) -> bool:
        """PDF에 텍스트 레이어가 존재하는지 감지합니다.

        처음 sample_pages 페이지를 샘플링하여 평균 문자 수로 판단합니다.

        Args:
            pdf_bytes: PDF 파일 바이트
            sample_pages: 샘플링할 최대 페이지 수

        Returns:
            bool: 텍스트 레이어가 있으면 True
        """
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            char_counts: list[int] = []

            with pdfplumber.open(pdf_stream) as pdf:
                pages_to_check = pdf.pages[:sample_pages]
                for page in pages_to_check:
                    text = page.extract_text() or ""
                    char_counts.append(len(text.strip()))

            if not char_counts:
                return False

            avg_chars = sum(char_counts) / len(char_counts)
            return avg_chars >= _MIN_CHARS_FOR_TEXT_LAYER

        except Exception as exc:
            logger.warning("[%s] 텍스트 레이어 감지 실패: %s", _ENGINE_NAME, exc)
            return False

    # ── 내부 메서드 ─────────────────────────────────────────────────────────

    def _parse_page(self, page: Page) -> PageOCRResult:
        """단일 pdfplumber Page 객체를 파싱합니다.

        Args:
            page: pdfplumber Page 인스턴스

        Returns:
            PageOCRResult: 파싱된 페이지 결과
        """
        page_width = float(page.width or 1.0)
        page_height = float(page.height or 1.0)

        # ── 표 추출 ────────────────────────────────────────────────────────
        tables = self._extract_tables(page, page_width, page_height)

        # 표 영역 bbox 집합 (블록 필터링에 사용)
        table_bboxes = [t["bbox"] for t in tables]

        # ── 텍스트 블록 추출 ────────────────────────────────────────────────
        blocks = self._extract_blocks(page, page_width, page_height, table_bboxes)

        # ── 전체 페이지 텍스트 (읽기 순서: y좌표 기준 정렬) ──────────────────
        full_text = self._build_page_text(blocks, tables, page_width, page_height)

        return PageOCRResult(
            page_number=page.page_number,
            text=full_text,
            blocks=blocks,
            tables=tables,
            confidence=1.0,  # 텍스트 PDF 신뢰도 최대값
        )

    def _extract_blocks(
        self,
        page: Page,
        page_width: float,
        page_height: float,
        table_bboxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        """pdfplumber words를 텍스트 블록으로 그룹화합니다.

        Args:
            page: pdfplumber Page
            page_width: 페이지 너비 (정규화 기준)
            page_height: 페이지 높이 (정규화 기준)
            table_bboxes: 표 바운딩 박스 목록 (제외 영역)

        Returns:
            list[dict]: 텍스트 블록 목록
        """
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        )
        if not words:
            return []

        # y좌표 기준 정렬 (읽기 순서 복원)
        words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))

        blocks: list[dict[str, Any]] = []
        current_line_words: list[dict] = []
        current_top: float | None = None
        line_tolerance = 5.0

        def flush_line() -> None:
            if not current_line_words:
                return
            line_text = " ".join(w["text"] for w in current_line_words)
            x0 = min(w["x0"] for w in current_line_words)
            y0 = min(w["top"] for w in current_line_words)
            x1 = max(w["x1"] for w in current_line_words)
            y1 = max(w["bottom"] for w in current_line_words)

            # 표 영역 내 단어 제외
            norm_bbox = self._normalize_bbox([x0, y0, x1, y1], page_width, page_height)
            if self._inside_table(norm_bbox, table_bboxes):
                return

            blocks.append(
                {
                    "text": line_text,
                    "bbox": norm_bbox,
                    "confidence": 1.0,
                    "block_type": "paragraph",
                }
            )

        for word in words_sorted:
            word_top = float(word["top"])
            if current_top is None or abs(word_top - current_top) > line_tolerance:
                flush_line()
                current_line_words = [word]
                current_top = word_top
            else:
                current_line_words.append(word)

        flush_line()
        return blocks

    def _extract_tables(
        self,
        page: Page,
        page_width: float,
        page_height: float,
    ) -> list[dict[str, Any]]:
        """pdfplumber extract_tables()로 표를 추출합니다.

        Args:
            page: pdfplumber Page
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            list[dict]: 표 목록
        """
        raw_tables = page.find_tables()
        tables: list[dict[str, Any]] = []

        for tbl_idx, tbl in enumerate(raw_tables):
            extracted = tbl.extract()
            if not extracted:
                continue

            # 표 바운딩 박스
            tbl_bbox = self._normalize_bbox(
                list(tbl.bbox), page_width, page_height
            )

            # 셀 구조화 (행·열 관계 보존)
            cells: list[list[dict[str, Any]]] = []
            for row_idx, row in enumerate(extracted):
                cell_row: list[dict[str, Any]] = []
                for col_idx, cell_text in enumerate(row):
                    cell_row.append(
                        {
                            "text": cell_text or "",
                            "row": row_idx,
                            "col": col_idx,
                            "bbox": [],  # 셀별 bbox는 필요 시 tbl.cells로 확장
                        }
                    )
                cells.append(cell_row)

            tables.append(
                {
                    "cells": cells,
                    "bbox": tbl_bbox,
                    "page": page.page_number,
                    "table_index": tbl_idx,
                }
            )

        return tables

    def _build_page_text(
        self,
        blocks: list[dict],
        tables: list[dict],
        page_width: float,
        page_height: float,
    ) -> str:
        """블록과 표를 읽기 순서로 결합하여 페이지 전체 텍스트를 생성합니다.

        Args:
            blocks: 텍스트 블록 목록
            tables: 표 목록
            page_width: 페이지 너비 (미사용, 확장용)
            page_height: 페이지 높이 (미사용, 확장용)

        Returns:
            str: 결합된 페이지 텍스트
        """
        items: list[dict[str, Any]] = []

        for block in blocks:
            items.append({"y0": block["bbox"][1], "text": block["text"]})

        for table in tables:
            table_text_lines: list[str] = []
            for row in table["cells"]:
                row_text = "\t".join(cell["text"] for cell in row)
                table_text_lines.append(row_text)
            items.append(
                {"y0": table["bbox"][1], "text": "\n".join(table_text_lines)}
            )

        items.sort(key=lambda x: x["y0"])
        return "\n".join(item["text"] for item in items).strip()

    @staticmethod
    def _normalize_bbox(
        bbox: list[float], page_width: float, page_height: float
    ) -> list[float]:
        """바운딩 박스를 [0, 1] 범위로 정규화합니다.

        Args:
            bbox: [x0, y0, x1, y1] 절대 좌표
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            list[float]: 정규화된 [x0, y0, x1, y1]
        """
        if page_width <= 0 or page_height <= 0:
            return bbox
        x0, y0, x1, y1 = bbox
        return [
            x0 / page_width,
            y0 / page_height,
            x1 / page_width,
            y1 / page_height,
        ]

    @staticmethod
    def _inside_table(
        bbox: list[float], table_bboxes: list[list[float]], threshold: float = 0.5
    ) -> bool:
        """주어진 bbox가 표 영역 내부에 있는지 확인합니다.

        Args:
            bbox: 확인할 블록의 정규화 bbox [x0, y0, x1, y1]
            table_bboxes: 표 bbox 목록
            threshold: 겹침 비율 임계값

        Returns:
            bool: 표 영역 내부이면 True
        """
        bx0, by0, bx1, by1 = bbox
        b_area = max((bx1 - bx0) * (by1 - by0), 1e-9)

        for tx0, ty0, tx1, ty1 in table_bboxes:
            ix0 = max(bx0, tx0)
            iy0 = max(by0, ty0)
            ix1 = min(bx1, tx1)
            iy1 = min(by1, ty1)
            if ix1 > ix0 and iy1 > iy0:
                overlap = (ix1 - ix0) * (iy1 - iy0)
                if overlap / b_area >= threshold:
                    return True
        return False
