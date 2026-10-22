"""Gemini LLM 클라이언트 래퍼"""
import json, re
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiClient:
    """
    Vertex AI Gemini 모델을 이용한 JSON 생성 클라이언트.
    API 미설정 시 None 반환 (에이전트들은 폴백 처리).
    """
    def __init__(self, project: str, location: str, model_name: str):
        self.project = project
        self.location = location
        self.model_name = model_name
        self._model = None
        self._init()
    
    def _init(self):
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            if self.project and self.project != "your-project-id":
                vertexai.init(project=self.project, location=self.location)
                self._model = GenerativeModel(self.model_name)
                print(f"[GeminiClient] 초기화 완료: {self.model_name}")
        except Exception as e:
            print(f"[GeminiClient] 초기화 실패, LLM 기능 비활성화: {e}")
    
    @property
    def is_available(self) -> bool:
        return self._model is not None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def generate_json(self, system: str, user: str) -> dict:
        """JSON 형식으로 응답 생성"""
        if not self._model:
            raise RuntimeError("LLM 미초기화")
        
        prompt = f"{system}\n\n{user}\n\n반드시 유효한 JSON만 반환하세요."
        
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(prompt)
        )
        text = response.text.strip()
        
        # JSON 추출
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)
