"""
app/config.py
=============
애플리케이션 설정 모듈.

pydantic BaseSettings를 이용해 .env 파일 및 환경 변수에서
설정값을 자동으로 로드합니다.
get_settings() 함수는 lru_cache로 싱글톤 패턴을 구현합니다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 전역 설정 클래스.

    환경 변수 또는 .env 파일에서 값을 자동으로 읽어 들입니다.
    필드 기본값은 .env.example의 예시 값을 참조합니다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Cloud ────────────────────────────────────────────────────────
    google_cloud_project: str = Field(default="", description="GCP 프로젝트 ID")
    google_cloud_location: str = Field(
        default="asia-northeast3", description="GCP 리전"
    )
    google_application_credentials: str = Field(
        default="", description="서비스 계정 키 파일 경로"
    )

    # ── Document AI ─────────────────────────────────────────────────────────
    document_ai_processor_id: str = Field(
        default="", description="Document AI 프로세서 ID"
    )
    document_ai_processor_location: str = Field(
        default="us", description="Document AI 프로세서 리전"
    )

    # ── Vertex AI ───────────────────────────────────────────────────────────
    vertex_ai_embedding_model: str = Field(
        default="text-multilingual-embedding-002",
        description="Vertex AI 임베딩 모델명",
    )
    vertex_ai_llm_model: str = Field(
        default="gemini-1.5-pro-002",
        description="Vertex AI LLM 모델명",
    )

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_db_path: str = Field(
        default="./storage/chroma_db", description="ChromaDB 저장 경로"
    )

    # ── Storage ─────────────────────────────────────────────────────────────
    storage_base_path: str = Field(
        default="./storage", description="스토리지 루트 경로"
    )
    originals_path: str = Field(
        default="./storage/originals", description="원본 파일 보관 경로"
    )
    ocr_results_path: str = Field(
        default="./storage/ocr_results", description="OCR 결과 저장 경로"
    )

    # ── App Settings ────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=50, description="업로드 최대 파일 크기 (MB)")
    max_pages: int = Field(default=200, description="처리 최대 페이지 수")
    chunk_min_tokens: int = Field(default=500, description="청크 최소 토큰 수")
    chunk_max_tokens: int = Field(default=900, description="청크 최대 토큰 수")
    chunk_overlap_tokens: int = Field(default=100, description="청크 오버랩 토큰 수")
    top_k_candidates: int = Field(
        default=10, description="유사도 후보 상위 K개"
    )
    top_n_retrieval: int = Field(default=50, description="검색 최대 반환 수")

    # ── 편의 프로퍼티 ────────────────────────────────────────────────────────
    @property
    def originals_dir(self) -> Path:
        """원본 파일 저장 디렉토리 Path 객체."""
        return Path(self.originals_path)

    @property
    def ocr_results_dir(self) -> Path:
        """OCR 결과 저장 디렉토리 Path 객체."""
        return Path(self.ocr_results_path)

    @property
    def chroma_db_dir(self) -> Path:
        """ChromaDB 디렉토리 Path 객체."""
        return Path(self.chroma_db_path)

    @property
    def max_file_size_bytes(self) -> int:
        """업로드 최대 파일 크기 (바이트)."""
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """필요한 스토리지 디렉토리를 모두 생성합니다."""
        for directory in [
            self.originals_dir,
            self.ocr_results_dir,
            self.chroma_db_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str):
        """대문자/소문자 속성 접근 호환 지원."""
        lower_name = name.lower()
        if lower_name in self.__dict__:
            return self.__dict__[lower_name]
        return super().__getattribute__(name)



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """싱글톤 Settings 인스턴스를 반환합니다.

    최초 호출 시 .env를 파싱하고 이후에는 캐시된 객체를 반환합니다.

    Returns:
        Settings: 전역 설정 객체
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
