"""
app/ocr/adapters/base.py
========================
OCR 어댑터 추상 베이스 클래스.

모든 OCR 엔진 어댑터(pdfplumber, Document AI, Vision OCR)가
구현해야 하는 공통 인터페이스를 정의합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PageOCRResult:
    """단일 페이지 OCR 결과 데이터 클래스.

    Attributes:
        page_number: 페이지 번호 (1-based)
        text: 해당 페이지의 전체 텍스트 (읽기 순서 복원 후)
        blocks: 텍스트 블록 목록. 각 블록은 다음 키를 포함:
                {text: str, bbox: [x0,y0,x1,y1], confidence: float, block_type: str}
        tables: 표 목록. 각 표는 다음 키를 포함:
                {cells: [[{text,row,col,bbox}]], bbox: [x0,y0,x1,y1], page: int}
        confidence: 페이지 전체 평균 신뢰도 (0.0 ~ 1.0)
        raw_response: 엔진별 원시 응답 (디버깅·감사용, 선택적)
        quality_flags: 페이지별 품질 경고 플래그 목록
    """

    page_number: int
    text: str
    blocks: list[dict]      # {text, bbox, confidence, block_type}
    tables: list[dict]      # {cells: [[{text, row, col, bbox}]], bbox, page}
    confidence: float
    raw_response: dict | None = None
    quality_flags: list[str] = field(default_factory=list)


class NotConfiguredError(Exception):
    """OCR 어댑터 설정이 누락된 경우 발생하는 예외."""

    def __init__(self, adapter_name: str, missing_config: str) -> None:
        self.adapter_name = adapter_name
        self.missing_config = missing_config
        super().__init__(
            f"[{adapter_name}] 설정이 누락되었습니다: {missing_config}. "
            f".env 파일을 확인하세요."
        )


class OCRProcessingError(Exception):
    """OCR 처리 중 발생하는 일반 예외."""

    def __init__(self, adapter_name: str, message: str, page: int | None = None) -> None:
        self.adapter_name = adapter_name
        self.page = page
        prefix = f"[{adapter_name}]"
        if page is not None:
            prefix += f" 페이지 {page}"
        super().__init__(f"{prefix}: {message}")


class BaseOCRAdapter(ABC):
    """OCR 어댑터 추상 베이스 클래스.

    모든 OCR 엔진 어댑터는 이 클래스를 상속하고
    추상 메서드를 구현해야 합니다.
    """

    @abstractmethod
    async def process_page(
        self, page_image_bytes: bytes, page_number: int
    ) -> PageOCRResult:
        """단일 페이지 이미지를 OCR 처리합니다.

        Args:
            page_image_bytes: 페이지 이미지 바이트 (PNG/JPEG)
            page_number: 페이지 번호 (1-based)

        Returns:
            PageOCRResult: 해당 페이지의 OCR 결과

        Raises:
            OCRProcessingError: OCR 처리 실패 시
            NotConfiguredError: 어댑터 설정 누락 시
        """
        ...

    @abstractmethod
    async def process_pdf(
        self, pdf_bytes: bytes, doc_id: str
    ) -> list[PageOCRResult]:
        """PDF 전체를 OCR 처리합니다.

        Args:
            pdf_bytes: PDF 파일 바이트
            doc_id: 문서 고유 식별자 (로깅·추적용)

        Returns:
            list[PageOCRResult]: 페이지별 OCR 결과 목록 (페이지 순서 보장)

        Raises:
            OCRProcessingError: OCR 처리 실패 시
            NotConfiguredError: 어댑터 설정 누락 시
        """
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """OCR 엔진 식별자 문자열.

        Returns:
            str: 예) "pdfplumber", "google-document-ai", "google-vision-ocr"
        """
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """OCR 모델 또는 어댑터 버전 문자열.

        Returns:
            str: 예) "0.11.x", "pretrained-v1.3"
        """
        ...
