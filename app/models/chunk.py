"""
app/models/chunk.py
===================
청크(Chunk) 도메인 모델.

기술명세서에 정의된 Chunk 스키마를 pydantic BaseModel로 구현합니다.
OCR 결과를 청킹하여 벡터 DB에 색인하기 전 중간 표현으로 사용됩니다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """청크 유형 열거형.

    Attributes:
        paragraph: 일반 텍스트 문단
        table: 표 전체
        table_row_group: 표의 행 그룹 (긴 표 분할 시)
        figure_caption: 그림·이미지 캡션
        list_item: 목록 항목
    """

    paragraph = "paragraph"
    table = "table"
    table_row_group = "table_row_group"
    figure_caption = "figure_caption"
    list_item = "list_item"


class SourceTextRef(BaseModel):
    """청크 원문 참조 정보.

    Attributes:
        page: 원본 페이지 번호 (1-based)
        block_id: OCR 블록 식별자
        bbox: 바운딩 박스 [x0, y0, x1, y1] (정규화 좌표)
        raw_text: 정규화 이전 원본 텍스트
    """

    page: int = Field(..., ge=1, description="원본 페이지 번호 (1-based)")
    block_id: str = Field(default="", description="OCR 블록 식별자")
    bbox: list[float] = Field(
        default_factory=list,
        description="바운딩 박스 [x0, y0, x1, y1] (정규화 좌표)",
    )
    raw_text: str = Field(default="", description="정규화 이전 원본 텍스트")


class ChunkModel(BaseModel):
    """청크 데이터 모델.

    OCR 결과에서 추출한 텍스트 단위로, 벡터 DB 색인 및 검색에 사용됩니다.

    Attributes:
        chunk_id: 청크 고유 식별자 (UUID)
        document_id: 소속 문서 ID
        section_path: 섹션 계층 경로 (예: ["1장", "1.2 요구사항"])
        chunk_type: 청크 유형
        text: 정규화된 청크 텍스트
        source_text_ref: 원문 위치 참조 정보
        page_start: 청크 시작 페이지 번호 (1-based)
        page_end: 청크 종료 페이지 번호 (1-based)
        keywords: 추출된 키워드 목록
        entities: 추출된 개체명 목록
        ocr_confidence: OCR 신뢰도 (0.0 ~ 1.0)
        quality_flags: 품질 경고 플래그 목록
        chunk_index: 문서 내 청크 순서 (0-based)
        chunk_hash: 청크 텍스트 SHA-256 해시 (중복 탐지용)
        chunker_version: 청킹 엔진 버전
    """

    chunk_id: str = Field(..., description="청크 고유 식별자 (UUID)")
    document_id: str = Field(..., description="소속 문서 ID")
    section_path: list[str] = Field(
        default_factory=list,
        description="섹션 계층 경로 (예: ['1장', '1.2 요구사항'])",
    )
    chunk_type: ChunkType = Field(
        default=ChunkType.paragraph, description="청크 유형"
    )
    text: str = Field(..., description="정규화된 청크 텍스트")
    source_text_ref: SourceTextRef | None = Field(
        default=None, description="원문 위치 참조 정보"
    )
    page_start: int = Field(default=1, ge=1, description="청크 시작 페이지 번호 (1-based)")
    page_end: int = Field(default=1, ge=1, description="청크 종료 페이지 번호 (1-based)")
    keywords: list[str] = Field(
        default_factory=list, description="추출된 키워드 목록"
    )
    entities: list[str] = Field(
        default_factory=list, description="추출된 개체명 목록"
    )
    ocr_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="OCR 신뢰도 (0.0 ~ 1.0)"
    )
    quality_flags: list[str] = Field(
        default_factory=list, description="품질 경고 플래그 목록"
    )
    chunk_index: int = Field(
        default=0, ge=0, description="문서 내 청크 순서 (0-based)"
    )
    chunk_hash: str = Field(
        default="", description="청크 텍스트 SHA-256 해시 (중복 탐지용)"
    )
    chunker_version: str = Field(
        default="1.0.0", description="청킹 엔진 버전"
    )
