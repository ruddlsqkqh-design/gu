"""Evidence Validation Agent - 근거 인용 검증"""

class EvidenceValidationAgent:
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def run(self, scored_candidates: list[dict]) -> list[dict]:
        """근거 검증 및 불확실성 추가"""
        validated = []
        for candidate in scored_candidates:
            v = candidate.copy()
            # 근거 일관성 확인
            matched = v.get("matched_passages", [])
            caveats = list(v.get("quality_and_caveats", []))
            
            if not matched:
                caveats.append("유사 구간이 검색되지 않았습니다. 수동 확인 필요.")
            
            if v.get("overall_similarity_score", 0) >= 70 and not self.llm:
                caveats.append("LLM 미설정으로 차원별 점수는 추정값입니다. 전문가 검토 필요.")
            
            # LLM 사용 시 차이점 분석
            if self.llm and matched:
                try:
                    v["differences"] = await self._analyze_differences(candidate)
                except Exception:
                    pass
            
            v["quality_and_caveats"] = caveats
            validated.append(v)
        return validated
    
    async def _analyze_differences(self, candidate: dict) -> list[str]:
        passages_text = str(candidate.get("matched_passages", [])[:3])
        dims = candidate.get("dimension_scores", {})
        
        prompt = f"""유사 구간과 차원별 점수를 보고 두 문서의 주요 차이점을 3개 이내로 나열하세요.
차원 점수: {dims}
유사 구간 정보: {passages_text}
반환: {{"differences": ["차이점1", "차이점2"]}}"""
        
        result = await self.llm.generate_json(system="분석 전문가입니다.", user=prompt)
        return result.get("differences", [])
