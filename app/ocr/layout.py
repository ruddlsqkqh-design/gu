"""
app/ocr/layout.py
=================
레이아웃 분석 및 섹션 트리 구축.

OCR 블록의 글자 크기·번호 패턴·위치 등을 분석하여
제목 계층 구조(섹션 트리)를 생성하고, 각 블록의 섹션 경로를 반환합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# 제목 번호 패턴 (우선순위 순)
_HEADING_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    (1, re.compile(r"^제\s*\d+\s*[장편절]\s")),           # 제1장, 제2편
    (1, re.compile(r"^\d+\.\s")),                         # 1. 제목
    (2, re.compile(r"^\d+\.\d+\s")),                      # 1.1 제목
    (2, re.compile(r"^[가나다라마바사아자차카타파하]\.\s")),  # 가. 나.
    (3, re.compile(r"^\d+\.\d+\.\d+\s")),                 # 1.1.1 제목
    (3, re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s")),             # ① ②
    (3, re.compile(r"^\([가나다라]\)\s")),                  # (가) (나)
    (4, re.compile(r"^[-•·∙▪▫]\s")),                      # 불릿 (4단계)
]

# 텍스트 PDF에서 제목 추정: 짧고 마침표 없는 줄
_SHORT_HEADING_MAX_LEN = 60
_SHORT_HEADING_RE = re.compile(r"[.。！!？?]$")

# 머리말/꼬리말 패턴
_HEADER_FOOTER_RE = re.compile(
    r"^(\d+\s*/\s*\d+|페이지\s*\d+|\d+\s*페이지|page\s*\d+)$",
    re.IGNORECASE,
)


@dataclass
class SectionNode:
    """섹션 트리 노드."""
    title: str
    level: int  # 1=최상위, 숫자 클수록 하위
    block_index: int  # 원본 블록 인덱스
    children: list["SectionNode"] = field(default_factory=list)


class LayoutAnalyzer:
    """OCR 블록 레이아웃 분석기.

    텍스트 블록 목록에서 제목을 탐지하고 섹션 트리를 구축합니다.
    """

    def classify_block_type(self, block: dict) -> str:
        """블록 유형을 분류합니다.

        Args:
            block: OCR 블록 dict {"text", "bbox", "confidence", "block_type"}

        Returns:
            str: "heading1" | "heading2" | "heading3" | "heading4"
                 | "header_footer" | "list_item" | "paragraph"
        """
        text = block.get("text", "").strip()
        if not text:
            return "paragraph"

        # 머리말/꼬리말 감지
        if _HEADER_FOOTER_RE.match(text):
            return "header_footer"

        # 번호 패턴으로 제목 탐지
        for level, pattern in _HEADING_PATTERNS:
            if pattern.match(text):
                if level <= 3:
                    return f"heading{level}"
                return "list_item"

        # 짧고 마침표 없는 줄 → 제목 후보 (2단계로 분류)
        if (
            len(text) <= _SHORT_HEADING_MAX_LEN
            and not _SHORT_HEADING_RE.search(text)
            and not text.startswith(" ")
        ):
            # 한국어 명사형 어미로 끝나면 제목으로 간주
            if re.search(r"[가-힣]{2,}(개요|목적|현황|방향|계획|전략|기술|시장|사업|배경|분석)$", text):
                return "heading2"

        return "paragraph"

    def build_section_tree(self, blocks: list[dict]) -> list[SectionNode]:
        """블록 목록에서 섹션 트리를 구축합니다.

        Args:
            blocks: OCR 블록 목록

        Returns:
            list[SectionNode]: 최상위 섹션 노드 목록
        """
        roots: list[SectionNode] = []
        # 레벨별 최근 노드 스택 {level: node}
        stack: dict[int, SectionNode] = {}

        for idx, block in enumerate(blocks):
            block_type = self.classify_block_type(block)
            if not block_type.startswith("heading"):
                continue

            level = int(block_type[-1])  # heading1 → 1
            node = SectionNode(
                title=block.get("text", "").strip(),
                level=level,
                block_index=idx,
            )

            if level == 1 or not stack:
                roots.append(node)
                stack = {level: node}
            else:
                # 부모 노드 탐색 (자신보다 낮은 레벨 중 가장 가까운 것)
                parent_level = max(
                    (lv for lv in stack if lv < level), default=None
                )
                if parent_level is not None:
                    stack[parent_level].children.append(node)
                else:
                    roots.append(node)

                # 현재 레벨 이상의 스택 엔트리 제거
                for lv in list(stack.keys()):
                    if lv >= level:
                        del stack[lv]
                stack[level] = node

        return roots

    def get_section_path(
        self, block_index: int, section_tree: list[SectionNode]
    ) -> list[str]:
        """특정 블록의 섹션 경로를 반환합니다.

        Args:
            block_index: 조회할 블록 인덱스
            section_tree: 섹션 트리 (build_section_tree 결과)

        Returns:
            list[str]: 섹션 경로 (예: ["1. 기술개요", "1.1 핵심기술"])
        """
        path: list[str] = []
        if self._find_path(block_index, section_tree, path):
            return path
        return []

    def build_block_section_map(
        self, blocks: list[dict], section_tree: list[SectionNode]
    ) -> dict[int, list[str]]:
        """모든 블록에 대한 섹션 경로 매핑을 생성합니다.

        연속 블록 스캔으로 O(n) 시간에 계산합니다.

        Args:
            blocks: OCR 블록 목록
            section_tree: 섹션 트리

        Returns:
            dict[int, list[str]]: {block_index: section_path}
        """
        # 제목 블록과 해당 섹션 경로를 순서대로 수집
        heading_entries: list[tuple[int, list[str]]] = []
        self._collect_headings(section_tree, [], heading_entries)
        heading_entries.sort(key=lambda x: x[0])

        result: dict[int, list[str]] = {}
        current_path: list[str] = []
        heading_iter = iter(heading_entries)
        next_heading = next(heading_iter, None)

        for idx in range(len(blocks)):
            # 다음 제목 블록에 도달했으면 경로 갱신
            while next_heading and next_heading[0] <= idx:
                current_path = next_heading[1]
                next_heading = next(heading_iter, None)
            result[idx] = list(current_path)

        return result

    # ── 내부 메서드 ─────────────────────────────────────────────────────────

    def _find_path(
        self,
        target_index: int,
        nodes: list[SectionNode],
        path: list[str],
    ) -> bool:
        """DFS로 블록 인덱스에 해당하는 섹션 경로를 탐색합니다."""
        for node in nodes:
            path.append(node.title)
            if node.block_index == target_index:
                return True
            if self._find_path(target_index, node.children, path):
                return True
            path.pop()
        return False

    def _collect_headings(
        self,
        nodes: list[SectionNode],
        current_path: list[str],
        result: list[tuple[int, list[str]]],
    ) -> None:
        """섹션 트리를 DFS 순회하며 (block_index, path) 목록을 수집합니다."""
        for node in nodes:
            path = current_path + [node.title]
            result.append((node.block_index, path))
            self._collect_headings(node.children, path, result)
