"""
app/ocr/chunker.py
==================
기술명세서 SPEC 1 §5 청킹 정책을 구현하는 청킹 엔진.

청킹 단위 우선순위:
  1. 문서 섹션 경계
  2. 하위 제목 / 기술 항목
  3. 문단 묶음
  4. 문장 경계
  5. 고정 길이 분할 (마지막 대안)

파라미터 기본값: 본문 500~900 토큰, 겹침 100 토큰.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    # tiktoken 미설치 시 글자 수 / 2 로 근사
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 2)

from app.models.chunk import ChunkModel, ChunkType, SourceTextRef
from app.ocr.adapters.base import PageOCRResult
from app.ocr.normalizer import normalize_text, extract_keywords_simple

_CHUNKER_VERSION = "1.0.0"

# 문장 경계 (마침표·느낌표·물음표 뒤 공백 또는 줄바꿈)
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass
class _TextBlock:
    """청킹 중간 단위."""
    text: str
    page: int
    bbox: list[float] = field(default_factory=list)
    block_id: str = ""
    section_path: list[str] = field(default_factory=list)
    block_type: str = "paragraph"  # paragraph | heading | table | list_item
    confidence: float = 1.0
    raw_text: str = ""


class ChunkingEngine:
    """기술명세서 청킹 정책 구현 엔진.

    Args:
        min_tokens: 청크 최소 토큰 수 (기본: 500)
        max_tokens: 청크 최대 토큰 수 (기본: 900)
        overlap_tokens: 인접 청크 겹침 토큰 수 (기본: 100)
    """

    def __init__(
        self,
        min_tokens: int = 500,
        max_tokens: int = 900,
        overlap_tokens: int = 100,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # ── 공개 메서드 ─────────────────────────────────────────────────────────

    def chunk_document(
        self,
        document_id: str,
        ocr_results: list[PageOCRResult],
        section_map: dict[int, list[str]] | None = None,
    ) -> list[ChunkModel]:
        """문서 전체를 청킹합니다.

        Args:
            document_id: 문서 ID
            ocr_results: 페이지별 OCR 결과 목록
            section_map: {(page, block_idx): section_path} 매핑 (없으면 빈 경로)

        Returns:
            list[ChunkModel]: 생성된 청크 목록
        """
        # 1. 표 청크 먼저 생성
        table_chunks = self._extract_table_chunks(document_id, ocr_results)

        # 2. 텍스트 블록 수집
        text_blocks = self._collect_text_blocks(ocr_results, section_map or {})

        # 3. 섹션 경계 기반 청킹 → 크기 초과 시 세분화
        text_chunks = self._chunk_text_blocks(document_id, text_blocks)

        # 4. 표 + 텍스트 청크를 페이지 순으로 병합
        all_chunks = sorted(
            table_chunks + text_chunks,
            key=lambda c: (c.page_start, c.chunk_index),
        )

        # 5. chunk_index 재부여
        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx

        return all_chunks

    # ── 내부: 표 청킹 ────────────────────────────────────────────────────────

    def _extract_table_chunks(
        self, document_id: str, ocr_results: list[PageOCRResult]
    ) -> list[ChunkModel]:
        """OCR 결과에서 표를 독립 청크로 변환합니다.

        기술명세서: 표 제목·열 헤더·행 값·단위·각주를 함께 포함,
        긴 표는 행 그룹 단위로 분할.
        """
        chunks: list[ChunkModel] = []
        chunk_idx = 0

        for page_result in ocr_results:
            for tbl in page_result.tables:
                tbl_chunks = self._chunk_table(
                    table=tbl,
                    document_id=document_id,
                    page=page_result.page_number,
                    start_idx=chunk_idx,
                )
                chunks.extend(tbl_chunks)
                chunk_idx += len(tbl_chunks)

        return chunks

    def _chunk_table(
        self,
        table: dict[str, Any],
        document_id: str,
        page: int,
        start_idx: int = 0,
    ) -> list[ChunkModel]:
        """단일 표를 청크로 변환합니다.

        첫 행은 열 헤더로 간주하고 모든 행 그룹에 포함시킵니다.
        """
        cells: list[list[dict]] = table.get("cells", [])
        if not cells:
            return []

        header_row = cells[0] if cells else []
        header_text = "| " + " | ".join(c.get("text", "") for c in header_row) + " |"
        separator = "| " + " | ".join("---" for _ in header_row) + " |"
        data_rows = cells[1:]

        # 행 그룹 분할 (max_tokens 기준)
        chunks: list[ChunkModel] = []
        current_rows: list[list[dict]] = []
        current_text = f"{header_text}\n{separator}"

        def flush_group(rows: list[list[dict]], idx: int) -> ChunkModel:
            row_lines = "\n".join(
                "| " + " | ".join(c.get("text", "") for c in row) + " |"
                for row in rows
            )
            full_text = f"{header_text}\n{separator}\n{row_lines}"
            return self._make_chunk(
                text=full_text,
                document_id=document_id,
                page=page,
                chunk_index=idx,
                chunk_type=ChunkType.table_row_group if chunks else ChunkType.table,
                bbox=table.get("bbox", []),
            )

        for row in data_rows:
            row_text = "| " + " | ".join(c.get("text", "") for c in row) + " |"
            tentative = f"{current_text}\n{row_text}"
            if _count_tokens(tentative) > self.max_tokens and current_rows:
                chunks.append(flush_group(current_rows, start_idx + len(chunks)))
                current_rows = [row]
                current_text = f"{header_text}\n{separator}\n{row_text}"
            else:
                current_rows.append(row)
                current_text = tentative

        if current_rows:
            chunks.append(flush_group(current_rows, start_idx + len(chunks)))
        elif not chunks:
            # 헤더만 있는 표
            chunks.append(flush_group([], start_idx))

        return chunks

    # ── 내부: 텍스트 청킹 ────────────────────────────────────────────────────

    def _collect_text_blocks(
        self,
        ocr_results: list[PageOCRResult],
        section_map: dict[int, list[str]],
    ) -> list[_TextBlock]:
        """OCR 결과에서 텍스트 블록을 수집합니다."""
        blocks: list[_TextBlock] = []
        global_idx = 0

        for page_result in ocr_results:
            for blk in page_result.blocks:
                raw = blk.get("text", "")
                normalized = normalize_text(raw)
                if not normalized:
                    global_idx += 1
                    continue

                blocks.append(
                    _TextBlock(
                        text=normalized,
                        page=page_result.page_number,
                        bbox=blk.get("bbox", []),
                        block_id=blk.get("block_id", f"p{page_result.page_number}_b{global_idx}"),
                        section_path=section_map.get(global_idx, []),
                        block_type=blk.get("block_type", "paragraph"),
                        confidence=blk.get("confidence", page_result.confidence),
                        raw_text=raw,
                    )
                )
                global_idx += 1

        return blocks

    def _chunk_text_blocks(
        self, document_id: str, blocks: list[_TextBlock]
    ) -> list[ChunkModel]:
        """텍스트 블록을 섹션 경계 기반으로 청킹합니다."""
        if not blocks:
            return []

        chunks: list[ChunkModel] = []
        buffer: list[_TextBlock] = []
        buffer_tokens = 0

        def flush(overlap_text: str = "") -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return

            combined = overlap_text + "\n\n".join(b.text for b in buffer)
            tokens = _count_tokens(combined)

            if tokens > self.max_tokens:
                # 문장 단위로 세분화
                sub_chunks = self._split_by_sentence(combined, buffer, document_id, len(chunks))
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    self._make_chunk(
                        text=combined,
                        document_id=document_id,
                        page=buffer[0].page,
                        chunk_index=len(chunks),
                        chunk_type=ChunkType.paragraph,
                        bbox=buffer[0].bbox,
                        section_path=buffer[0].section_path,
                        confidence=min(b.confidence for b in buffer),
                        raw_text=buffer[0].raw_text,
                    )
                )
            buffer = []
            buffer_tokens = 0

        prev_section: list[str] = []

        for block in blocks:
            # 제목 블록 → 섹션 경계: 현재 버퍼를 flush
            if block.block_type.startswith("heading"):
                flush()
                prev_section = block.section_path
                # 제목 자체는 다음 블록들과 함께 묶이도록 버퍼에 추가
                buffer.append(block)
                buffer_tokens += _count_tokens(block.text)
                continue

            # 섹션 경로 변경 → 경계
            if block.section_path and block.section_path != prev_section:
                flush()
                prev_section = block.section_path

            block_tokens = _count_tokens(block.text)

            # 버퍼가 최대 크기에 근접하면 flush
            if buffer_tokens + block_tokens > self.max_tokens and buffer_tokens >= self.min_tokens:
                # 겹침: 마지막 블록의 앞부분을 overlap으로 유지
                overlap = self._get_overlap_text(buffer)
                flush(overlap)

            buffer.append(block)
            buffer_tokens += block_tokens

        flush()
        return chunks

    def _split_by_sentence(
        self,
        text: str,
        blocks: list[_TextBlock],
        document_id: str,
        start_idx: int,
    ) -> list[ChunkModel]:
        """텍스트를 문장 단위로 분할합니다."""
        sentences = _SENT_BOUNDARY_RE.split(text)
        sub_chunks: list[ChunkModel] = []
        current: list[str] = []
        current_tokens = 0

        ref_block = blocks[0] if blocks else None

        def flush_sentences() -> None:
            if not current:
                return
            combined = " ".join(current)
            sub_chunks.append(
                self._make_chunk(
                    text=combined,
                    document_id=document_id,
                    page=ref_block.page if ref_block else 1,
                    chunk_index=start_idx + len(sub_chunks),
                    chunk_type=ChunkType.paragraph,
                    bbox=ref_block.bbox if ref_block else [],
                    section_path=ref_block.section_path if ref_block else [],
                    confidence=ref_block.confidence if ref_block else 1.0,
                )
            )
            current.clear()

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            t = _count_tokens(sent)
            if current_tokens + t > self.max_tokens and current_tokens > 0:
                flush_sentences()
                current_tokens = 0
            current.append(sent)
            current_tokens += t

        flush_sentences()
        return sub_chunks

    def _get_overlap_text(self, buffer: list[_TextBlock]) -> str:
        """버퍼 끝에서 overlap_tokens만큼의 텍스트를 반환합니다."""
        overlap_text = ""
        for block in reversed(buffer):
            candidate = block.text + "\n\n" + overlap_text if overlap_text else block.text
            if _count_tokens(candidate) > self.overlap_tokens:
                break
            overlap_text = candidate
        return overlap_text + "\n\n" if overlap_text else ""

    # ── 팩토리 ──────────────────────────────────────────────────────────────

    def _make_chunk(
        self,
        text: str,
        document_id: str,
        page: int,
        chunk_index: int,
        chunk_type: ChunkType = ChunkType.paragraph,
        bbox: list[float] | None = None,
        section_path: list[str] | None = None,
        confidence: float = 1.0,
        raw_text: str = "",
        block_id: str = "",
    ) -> ChunkModel:
        """ChunkModel 인스턴스를 생성합니다."""
        normalized = normalize_text(text)
        chunk_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        keywords = extract_keywords_simple(normalized, top_n=10)

        return ChunkModel(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            section_path=section_path or [],
            chunk_type=chunk_type,
            text=normalized,
            source_text_ref=SourceTextRef(
                page=page,
                block_id=block_id,
                bbox=bbox or [],
                raw_text=raw_text or text,
            ),
            page_start=page,
            page_end=page,
            keywords=keywords,
            entities=[],
            ocr_confidence=confidence,
            quality_flags=[],
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
            chunker_version=_CHUNKER_VERSION,
        )
