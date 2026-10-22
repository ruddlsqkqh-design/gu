"""
app/ocr/quality.py
==================
OCR 결과 및 청크 품질 검사기.

기술명세서 OCR-008 요구사항:
- 빈 페이지, 낮은 신뢰도, 깨진 문자, 표 추출 실패 탐지
- 재처리 경로 제공
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.chunk import ChunkModel
from app.ocr.adapters.base import PageOCRResult
from app.ocr.normalizer import has_broken_chars

# 품질 임계값
_LOW_CONFIDENCE_THRESHOLD = 0.6
_MIN_TEXT_LENGTH = 10       # 최소 텍스트 길이 (빈 페이지 판단)
_MIN_CHUNK_TOKENS = 30      # 너무 짧은 청크 경고
_MAX_BROKEN_RATIO = 0.05    # 깨진 문자 비율 임계값 (5%)


@dataclass
class DocumentQualitySummary:
    """문서 전체 품질 요약."""
    total_pages: int
    empty_pages: int
    low_confidence_pages: int
    broken_char_pages: int
    total_chunks: int
    short_chunks: int
    missing_ref_chunks: int
    needs_review: bool
    flags: list[str]

    def to_dict(self) -> dict:
        return {
            "total_pages": self.total_pages,
            "empty_pages": self.empty_pages,
            "low_confidence_pages": self.low_confidence_pages,
            "broken_char_pages": self.broken_char_pages,
            "total_chunks": self.total_chunks,
            "short_chunks": self.short_chunks,
            "missing_ref_chunks": self.missing_ref_chunks,
            "needs_review": self.needs_review,
            "flags": self.flags,
        }


class QualityChecker:
    """OCR 품질 검사기.

    Args:
        low_confidence_threshold: 낮은 신뢰도 판단 임계값 (기본: 0.6)
        min_text_length: 빈 페이지 판단 최소 글자 수 (기본: 10)
    """

    def __init__(
        self,
        low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
        min_text_length: int = _MIN_TEXT_LENGTH,
    ) -> None:
        self.low_confidence_threshold = low_confidence_threshold
        self.min_text_length = min_text_length

    def check_page(self, result: PageOCRResult) -> list[str]:
        """단일 페이지 OCR 결과를 검사합니다.

        Args:
            result: 페이지 OCR 결과

        Returns:
            list[str]: 감지된 품질 플래그 목록
                - "empty_page": 텍스트가 거의 없음
                - "low_confidence": OCR 신뢰도 낮음
                - "broken_chars": 깨진 문자 감지
                - "table_extraction_failed": 표 추출 실패
                - "no_blocks": 블록이 없음
        """
        flags: list[str] = []

        # 이미 플래그된 경우 유지
        if result.quality_flags:
            flags.extend(result.quality_flags)

        # 1. 빈 페이지
        text = result.text.strip()
        if len(text) < self.min_text_length:
            flags.append("empty_page")

        # 2. 낮은 신뢰도
        if 0.0 <= result.confidence < self.low_confidence_threshold:
            flags.append("low_confidence")

        # 3. 깨진 문자
        if text and has_broken_chars(text):
            broken_count = len(re.findall(r"[\uFFFD\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", text))
            ratio = broken_count / max(len(text), 1)
            if ratio > _MAX_BROKEN_RATIO:
                flags.append("broken_chars")

        # 4. 블록 없음
        if not result.blocks and text:
            flags.append("no_blocks")

        # 5. 표 파싱 오류 (셀이 모두 빈 경우)
        for tbl in result.tables:
            all_cells = [c.get("text", "") for row in tbl.get("cells", []) for c in row]
            if all_cells and all(not c.strip() for c in all_cells):
                flags.append("table_extraction_failed")
                break

        return list(set(flags))  # 중복 제거

    def check_chunk(self, chunk: ChunkModel) -> list[str]:
        """단일 청크를 검사합니다.

        Args:
            chunk: 청크 모델

        Returns:
            list[str]: 감지된 품질 플래그 목록
                - "empty_chunk": 텍스트가 없음
                - "short_chunk": 텍스트가 너무 짧음
                - "missing_source_ref": 원문 위치 정보 없음
                - "low_ocr_confidence": OCR 신뢰도 낮음
                - "broken_chars": 깨진 문자 감지
        """
        flags: list[str] = []

        # 1. 빈 청크
        if not chunk.text.strip():
            flags.append("empty_chunk")
            return flags

        # 2. 너무 짧은 청크
        token_approx = max(1, len(chunk.text) // 2)
        if token_approx < _MIN_CHUNK_TOKENS:
            flags.append("short_chunk")

        # 3. 원문 위치 정보 없음
        if not chunk.source_text_ref or not chunk.source_text_ref.bbox:
            flags.append("missing_source_ref")

        # 4. 낮은 OCR 신뢰도
        if 0.0 <= chunk.ocr_confidence < self.low_confidence_threshold:
            flags.append("low_ocr_confidence")

        # 5. 깨진 문자
        if has_broken_chars(chunk.text):
            flags.append("broken_chars")

        return flags

    def check_document(
        self,
        ocr_results: list[PageOCRResult],
        chunks: list[ChunkModel],
    ) -> DocumentQualitySummary:
        """문서 전체 품질을 검사합니다.

        Args:
            ocr_results: 페이지별 OCR 결과
            chunks: 청크 목록

        Returns:
            DocumentQualitySummary: 문서 품질 요약
        """
        total_pages = len(ocr_results)
        empty_pages = 0
        low_conf_pages = 0
        broken_pages = 0

        for result in ocr_results:
            flags = self.check_page(result)
            if "empty_page" in flags:
                empty_pages += 1
            if "low_confidence" in flags:
                low_conf_pages += 1
            if "broken_chars" in flags:
                broken_pages += 1

        total_chunks = len(chunks)
        short_chunks = 0
        missing_ref = 0

        for chunk in chunks:
            flags = self.check_chunk(chunk)
            if "short_chunk" in flags or "empty_chunk" in flags:
                short_chunks += 1
            if "missing_source_ref" in flags:
                missing_ref += 1

        # 전체 품질 플래그
        doc_flags: list[str] = []
        if empty_pages > total_pages * 0.2:
            doc_flags.append(f"high_empty_page_rate: {empty_pages}/{total_pages}")
        if low_conf_pages > 0:
            doc_flags.append(f"low_confidence_pages: {low_conf_pages}")
        if broken_pages > 0:
            doc_flags.append(f"broken_char_pages: {broken_pages}")
        if total_chunks == 0:
            doc_flags.append("no_chunks_generated")

        needs_review = bool(doc_flags) or low_conf_pages > total_pages * 0.1

        return DocumentQualitySummary(
            total_pages=total_pages,
            empty_pages=empty_pages,
            low_confidence_pages=low_conf_pages,
            broken_char_pages=broken_pages,
            total_chunks=total_chunks,
            short_chunks=short_chunks,
            missing_ref_chunks=missing_ref,
            needs_review=needs_review,
            flags=doc_flags,
        )
