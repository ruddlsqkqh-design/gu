"""Classification Agent - 기술 분야 분류"""

TECH_TAXONOMY = {
    "인공지능/머신러닝": ["AI", "ML", "딥러닝", "신경망", "학습", "모델", "추론"],
    "자연어처리": ["NLP", "OCR", "텍스트", "언어", "번역", "요약", "챗봇"],
    "컴퓨터비전": ["이미지", "영상", "인식", "검출", "분류", "객체"],
    "빅데이터/분석": ["데이터", "분석", "처리", "파이프라인", "ETL"],
    "클라우드/인프라": ["클라우드", "서버", "API", "마이크로서비스", "컨테이너"],
    "보안/인증": ["보안", "암호", "인증", "접근", "방화벽"],
    "기타": []
}

class ClassificationAgent:
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def run(self, tech_profile: dict) -> dict:
        text = str(tech_profile)
        
        if self.llm:
            try:
                prompt = f"""기술 프로필을 보고 기술 분야를 분류하세요:\n{text[:1000]}\n
반환 형식: {{"primary_category": "분야명", "secondary_categories": ["분야1"], "confidence": 0.85, "reasoning": "이유"}}"""
                return await self.llm.generate_json(system="기술 분류 전문가입니다.", user=prompt)
            except Exception as e:
                print(f"[Classification] LLM 오류: {e}")
        
        return self._rule_based(text)
    
    def _rule_based(self, text: str) -> dict:
        scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in TECH_TAXONOMY.items()}
        primary = max(scores, key=lambda x: scores[x])
        secondary = [c for c, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[1:3] if s > 0]
        return {
            "primary_category": primary,
            "secondary_categories": secondary,
            "confidence": min(0.5 + scores[primary] * 0.1, 0.9),
            "reasoning": "규칙 기반 분류"
        }
