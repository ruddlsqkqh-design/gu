"""
app/models/document.py
======================
문서(Document) 도메인 모델.

기술명세서에 정의된 Document 스키마를 pydantic BaseModel로 구현합니다.
OCR 파이프라인 전반에서 문서 상태 및 메타데이터를 추적하는 데 사용됩니다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """문서 처리 상태 열거형.

    Attributes:
        QUEUED: 처리 대기 중
        RUNNING: OCR 파이프라인 처리 중
        SUCCEEDED: 모든 페이지 처리 완료
        PARTIAL: 일부 페이지 처리 실패 (부분 성공)
        FAILED: 처리 전체 실패
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class DocumentModel(BaseModel):
    """문서 메타데이터 및 처리 상태 모델.

    OCR 파이프라인에서 생성·업데이트되며 JSON 파일로 영속화됩니다.

    Attributes:
        document_id: 문서 고유 식별자 (UUID)
        tenant_id: 테넌트(조직) 식별자
        source_uri: 원본 파일 경로 또는 URL
        original_filename: 업로드된 원본 파일명
        file_hash: SHA-256 파일 해시 (멱등성 처리에 사용)
        page_count: 총 페이지 수
        status: 현재 처리 상태
        ocr_engine: 사용된 OCR 엔진명
        model_version: OCR 모델/어댑터 버전
        created_at: 문서 생성 타임스탬프 (UTC)
        updated_at: 마지막 업데이트 타임스탬프 (UTC)
        quality_summary: 문서 전체 품질 요약 정보
        error_message: 실패 시 오류 메시지
    """

    document_id: str = Field(..., description="문서 고유 식별자 (UUID)")
    tenant_id: str = Field(default="default", description="테넌트 식별자")
    source_uri: str = Field(..., description="원본 파일 경로 또는 URI")
    original_filename: str = Field(default="", description="업로드된 원본 파일명")
    file_hash: str = Field(..., description="SHA-256 파일 해시값")
    page_count: int = Field(default=0, ge=0, description="총 페이지 수")
    status: DocumentStatus = Field(
        default=DocumentStatus.QUEUED, description="문서 처리 상태"
    )
    ocr_engine: str = Field(default="", description="사용된 OCR 엔진명")
    model_version: str = Field(default="", description="OCR 모델/어댑터 버전")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="문서 생성 타임스탬프 (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="마지막 업데이트 타임스탬프 (UTC)",
    )
    quality_summary: dict[str, Any] = Field(
        default_factory=dict, description="문서 전체 품질 요약 정보"
    )
    error_message: str | None = Field(default=None, description="실패 시 오류 메시지")

    def mark_running(self, ocr_engine: str = "", model_version: str = "") -> None:
        """문서 상태를 RUNNING으로 변경합니다."""
        self.status = DocumentStatus.RUNNING
        if ocr_engine:
            self.ocr_engine = ocr_engine
        if model_version:
            self.model_version = model_version
        self.updated_at = datetime.now(timezone.utc)

    def mark_succeeded(
        self,
        page_count: int,
        quality_summary: dict[str, Any] | None = None,
    ) -> None:
        """문서 상태를 SUCCEEDED로 변경합니다."""
        self.status = DocumentStatus.SUCCEEDED
        self.page_count = page_count
        if quality_summary:
            self.quality_summary = quality_summary
        self.updated_at = datetime.now(timezone.utc)

    def mark_partial(
        self,
        page_count: int,
        quality_summary: dict[str, Any] | None = None,
    ) -> None:
        """문서 상태를 PARTIAL(부분 성공)로 변경합니다."""
        self.status = DocumentStatus.PARTIAL
        self.page_count = page_count
        if quality_summary:
            self.quality_summary = quality_summary
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error_message: str) -> None:
        """문서 상태를 FAILED로 변경합니다."""
        self.status = DocumentStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)
