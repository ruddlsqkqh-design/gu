"""
app/ocr/adapters/vision_ocr.py
================================
Google Cloud Vision OCR 어댑터.

스캔 이미지 또는 이미지 기반 PDF 페이지에서 텍스트를 추출합니다.
document_text_detection API를 사용하며 블록, 단락, 단어 구조를 보존합니다.
API 설정이 없을 경우 NotConfiguredError를 발생시켜 graceful degradation을
지원합니다.

엔진명: google-vision-ocr
"""

from __future__ import annotations

import asyncio
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

_ENGINE_NAME = "google-vision-ocr"
_MAX_RETRY_ATTEMPTS = 3
_RETRY_WAIT_MIN = 2
_RETRY_WAIT_MAX = 10

_executor = ThreadPoolExecutor(max_workers=4)


class VisionOCRAdapter(BaseOCRAdapter):
    """Google Cloud Vision API OCR 어댑터.

    주로 스캔된 이미지 페이지에 사용됩니다.
    document_text_detection을 이용해 단락·블록 구조를 유지하며 텍스트를 추출합니다.

    Args:
        project_id: GCP 프로젝트 ID (클라이언트 초기화에만 사용)
    """

    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id
        self._client: Any | None = None
        self._configured = False
        self._init_client()

    def _init_client(self) -> None:
        """Vision API 클라이언트를 초기화합니다."""
        try:
            from google.cloud import vision  # type: ignore

            self._client = vision.ImageAnnotatorClient()
            self._configured = True
            logger.info("[%s] Vision OCR 클라이언트 초기화 완료", _ENGINE_NAME)
        except ImportError as exc:
            logger.warning(
                "[%s] google-cloud-vision 패키지 미설치: %s", _ENGINE_NAME, exc
            )
        except Exception as exc:
            logger.warning("[%s] 클라이언트 초기화 실패: %s", _ENGINE_NAME, exc)

    @property
    def engine_name(self) -> str:
        """OCR 엔진 식별자."""
        return _ENGINE_NAME

    @property
    def model_version(self) -> str:
        """Vision OCR 버전 (API 버전 고정)."""
        return "vision-v1"

    def _check_configured(self) -> None:
        """설정 완료 여부를 확인합니다.

        Raises:
            NotConfiguredError: 클라이언트가 초기화되지 않은 경우
        """
        if not self._configured or self._client is None:
            raise NotConfiguredError(
                _ENGINE_NAME,
                "GOOGLE_APPLICATION_CREDENTIALS 또는 google-cloud-vision 패키지",
            )

    async def process_page(
        self, page_image_bytes: bytes, page_number: int
    ) -> PageOCRResult:
        """단일 페이지 이미지를 Vision OCR로 처리합니다.

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
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor, self._detect_text_sync, page_image_bytes, page_number
            )
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
        """PDF를 페이지 이미지로 분리한 후 Vision OCR을 적용합니다.

        PDF는 PyMuPDF(fitz)로 래스터화하여 페이지별 이미지로 변환합니다.

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
        logger.info("[%s] 문서 %s: Vision OCR 처리 시작", _ENGINE_NAME, doc_id)

        try:
            import fitz  # PyMuPDF  # type: ignore
        except ImportError as exc:
            raise OCRProcessingError(
                _ENGINE_NAME, "PyMuPDF(fitz) 패키지가 필요합니다."
            ) from exc

        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            results: list[PageOCRResult] = []

            tasks = []
            for page_index in range(len(pdf_doc)):
                page = pdf_doc[page_index]
                # 150 DPI로 래스터화
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
                tasks.append((image_bytes, page_index + 1))

            pdf_doc.close()

            # 순차 처리 (Vision API 할당량 보호)
            for image_bytes, page_number in tasks:
                try:
                    result = await self.process_page(image_bytes, page_number)
                    results.append(result)
                except Exception as exc:
                    logger.warning(
                        "[%s] 문서 %s, 페이지 %d 처리 실패: %s",
                        _ENGINE_NAME,
                        doc_id,
                        page_number,
                        exc,
                    )
                    results.append(
                        PageOCRResult(
                            page_number=page_number,
                            text="",
                            blocks=[],
                            tables=[],
                            confidence=0.0,
                            quality_flags=["vision_ocr_error"],
                        )
                    )

            logger.info(
                "[%s] 문서 %s: %d 페이지 처리 완료", _ENGINE_NAME, doc_id, len(results)
            )
            return results

        except (NotConfiguredError, OCRProcessingError):
            raise
        except Exception as exc:
            raise OCRProcessingError(
                _ENGINE_NAME, f"PDF Vision OCR 처리 실패 (doc_id={doc_id}): {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=_RETRY_WAIT_MIN, max=_RETRY_WAIT_MAX),
        reraise=True,
    )
    def _detect_text_sync(
        self, image_bytes: bytes, page_number: int
    ) -> PageOCRResult:
        """Vision API document_text_detection 동기 호출.

        Args:
            image_bytes: 페이지 이미지 바이트
            page_number: 페이지 번호

        Returns:
            PageOCRResult: 변환된 OCR 결과
        """
        from google.cloud import vision  # type: ignore

        image = vision.Image(content=image_bytes)
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            raise OCRProcessingError(
                _ENGINE_NAME,
                f"Vision API 오류: {response.error.message}",
                page=page_number,
            )

        annotation = response.full_text_annotation
        if not annotation:
            return PageOCRResult(
                page_number=page_number,
                text="",
                blocks=[],
                tables=[],
                confidence=0.0,
                quality_flags=["empty_response"],
            )

        return self._convert_annotation(annotation, page_number)

    def _convert_annotation(
        self, annotation: Any, page_number: int
    ) -> PageOCRResult:
        """Vision API FullTextAnnotation을 PageOCRResult로 변환합니다.

        Args:
            annotation: Vision API FullTextAnnotation 객체
            page_number: 페이지 번호

        Returns:
            PageOCRResult: 변환된 OCR 결과
        """
        blocks: list[dict[str, Any]] = []
        confidences: list[float] = []
        full_text = annotation.text or ""

        for page in annotation.pages:
            page_width = float(page.width) if page.width else 1.0
            page_height = float(page.height) if page.height else 1.0

            for block in page.blocks:
                block_conf = float(block.confidence) if block.confidence else 0.5
                confidences.append(block_conf)

                block_text_parts: list[str] = []
                for paragraph in block.paragraphs:
                    para_text_parts: list[str] = []
                    for word in paragraph.words:
                        word_text = "".join(
                            symbol.text for symbol in word.symbols
                        )
                        para_text_parts.append(word_text)
                    block_text_parts.append(" ".join(para_text_parts))

                block_text = "\n".join(block_text_parts).strip()
                if not block_text:
                    continue

                bbox = self._convert_vision_bbox(
                    block.bounding_box, page_width, page_height
                )
                blocks.append(
                    {
                        "text": block_text,
                        "bbox": bbox,
                        "confidence": block_conf,
                        "block_type": self._map_block_type(block.block_type),
                    }
                )

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return PageOCRResult(
            page_number=page_number,
            text=full_text.strip(),
            blocks=blocks,
            tables=[],  # Vision OCR은 표 구조를 직접 제공하지 않음
            confidence=avg_conf,
        )

    @staticmethod
    def _convert_vision_bbox(
        bounding_box: Any, page_width: float, page_height: float
    ) -> list[float]:
        """Vision API BoundingPoly를 정규화된 [x0, y0, x1, y1]로 변환합니다.

        Args:
            bounding_box: Vision API BoundingPoly 객체
            page_width: 페이지 너비
            page_height: 페이지 높이

        Returns:
            list[float]: 정규화된 bbox
        """
        if not bounding_box or not bounding_box.vertices:
            return [0.0, 0.0, 1.0, 1.0]

        vertices = bounding_box.vertices
        xs = [v.x / page_width for v in vertices if v.x is not None]
        ys = [v.y / page_height for v in vertices if v.y is not None]

        if not xs or not ys:
            return [0.0, 0.0, 1.0, 1.0]

        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _map_block_type(block_type: Any) -> str:
        """Vision API BlockType을 문자열로 변환합니다.

        Args:
            block_type: Vision API BlockType 열거형 값

        Returns:
            str: 블록 유형 문자열
        """
        type_map = {
            0: "unknown",
            1: "paragraph",
            2: "table",
            3: "picture",
            4: "ruler",
            5: "barcode",
        }
        return type_map.get(int(block_type), "paragraph")
