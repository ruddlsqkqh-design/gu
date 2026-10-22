"""
app/ocr/normalizer.py
=====================
OCR 결과 텍스트 정규화 유틸리티.

유니코드 정규화, 전각→반각 변환, 공백/줄바꿈 정리,
하이픈 줄바꿈 연결 처리, 머리말/꼬리말 탐지를 제공합니다.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


# 전각→반각 변환 테이블 (U+FF01~U+FF5E → U+0021~U+007E)
_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E

# 하이픈 줄바꿈 패턴: "단어-\n다음단어"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# 과도한 공백 패턴
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# 깨진 문자 탐지 패턴 (대체 문자 U+FFFD, 제어 문자 등)
_BROKEN_CHAR_RE = re.compile(r"[\uFFFD\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def normalize_text(raw: str) -> str:
    """OCR 원본 텍스트를 정규화합니다.

    처리 순서:
    1. 유니코드 NFC 정규화
    2. 전각 문자 → 반각 변환
    3. 하이픈 줄바꿈 연결 (예: "단어-\\n다음" → "단어-다음")
    4. 과도한 공백/탭 제거
    5. 연속 줄바꿈 압축 (3개 이상 → 2개)
    6. 앞뒤 공백 제거

    Args:
        raw: 정규화 전 원본 텍스트

    Returns:
        str: 정규화된 텍스트
    """
    if not raw:
        return ""

    # 1. NFC 정규화 (합성 형식)
    text = unicodedata.normalize("NFC", raw)

    # 2. 전각 → 반각
    text = _fullwidth_to_halfwidth(text)

    # 3. 하이픈 줄바꿈 연결
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    # 4. 과도한 공백 정리 (탭 포함)
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 5. 과도한 줄바꿈 압축
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 6. 앞뒤 공백 제거
    return text.strip()


def has_broken_chars(text: str) -> bool:
    """텍스트에 깨진 문자(대체 문자, 제어 문자)가 있는지 확인합니다.

    Args:
        text: 검사할 텍스트

    Returns:
        bool: 깨진 문자가 있으면 True
    """
    return bool(_BROKEN_CHAR_RE.search(text))


def detect_header_footer(text: str, page_texts: list[str], min_repeat: int = 3) -> bool:
    """텍스트가 여러 페이지에 반복 등장하는 머리말/꼬리말인지 감지합니다.

    Args:
        text: 감지할 텍스트 (정규화 후)
    page_texts: 비교할 페이지 텍스트 목록
        min_repeat: 반복 등장 최소 횟수 (기본값: 3)

    Returns:
        bool: 머리말/꼬리말로 판단되면 True
    """
    if not text or not page_texts:
        return False

    normalized = text.strip()
    if len(normalized) < 3:
        return True  # 너무 짧은 텍스트는 페이지 번호 등으로 간주

    count = sum(1 for pt in page_texts if normalized in pt)
    return count >= min_repeat


def extract_keywords_simple(text: str, top_n: int = 20) -> list[str]:
    """텍스트에서 간단한 빈도 기반 키워드를 추출합니다.

    정규식 기반 한국어 명사구 추출 (형태소 분석기 없이).
    LLM/형태소 분석기가 없을 때 폴백으로 사용됩니다.

    Args:
        text: 입력 텍스트
        top_n: 반환할 최대 키워드 수

    Returns:
        list[str]: 빈도 기준 상위 키워드 목록
    """
    # 한국어 2글자 이상 연속 명사구 + 영문 단어
    korean = re.findall(r"[가-힣]{2,}", text)
    english = re.findall(r"[A-Za-z]{3,}", text)

    # 불용어 제거
    stopwords = {
        "이다", "있다", "하다", "되다", "이며", "이고", "으로", "에서", "에게",
        "부터", "까지", "위해", "통해", "같은", "이런", "이와", "또한", "그리고",
        "하지만", "그러나", "따라서", "the", "and", "for", "that", "this", "with",
    }

    words = [w for w in korean + english if w.lower() not in stopwords]
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


# ── 내부 함수 ────────────────────────────────────────────────────────────────

def _fullwidth_to_halfwidth(text: str) -> str:
    """전각 ASCII 문자를 반각으로 변환합니다.

    U+FF01 (！) ~ U+FF5E (～) 범위를 U+0021 (~) 으로 변환.
    전각 공백(U+3000)은 일반 공백으로 변환.

    Args:
        text: 변환할 텍스트

    Returns:
        str: 변환된 텍스트
    """
    result: list[str] = []
    for ch in text:
        cp = ord(ch)
        if _FULLWIDTH_START <= cp <= _FULLWIDTH_END:
            result.append(chr(cp - _FULLWIDTH_OFFSET))
        elif ch == "\u3000":  # 전각 공백
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result)
