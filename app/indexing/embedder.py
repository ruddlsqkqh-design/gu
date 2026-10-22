"""Vertex AI 임베딩 생성기"""
import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingService:
    """
    Vertex AI text-multilingual-embedding-002를 사용해 청크 텍스트를 임베딩.
    API 미설정 시 더미 임베딩(768차원 랜덤) 반환 (개발/테스트용).
    """
    def __init__(self, project: str, location: str, model_name: str):
        self.project = project
        self.location = location
        self.model_name = model_name
        self._client = None
        self._dimension = 768
        self._initialized = False
        self._init_client()
    
    def _init_client(self):
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            if self.project and self.project != "your-project-id":
                vertexai.init(project=self.project, location=self.location)
                self._client = TextEmbeddingModel.from_pretrained(self.model_name)
                self._initialized = True
        except Exception as e:
            print(f"[EmbeddingService] Vertex AI 초기화 실패, 더미 모드 사용: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 임베딩 벡터 리스트로 변환"""
        if not self._initialized:
            return self._dummy_embeddings(texts)
        
        # Vertex AI API 호출 (배치 처리, 최대 250개)
        results = []
        batch_size = 250
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self._client.get_embeddings(batch)
            results.extend([e.values for e in embeddings])
        return results
    
    def _dummy_embeddings(self, texts: list[str]) -> list[list[float]]:
        import hashlib, numpy as np
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.random(self._dimension).tolist()
            results.append(vec)
        return results
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_version(self) -> str:
        return self.model_name if self._initialized else f"{self.model_name}(dummy)"
