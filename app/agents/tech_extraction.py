"""Technology Extraction Agent - 기술 요소·구성·성능 지표 추출"""

class TechnologyExtractionAgent:
    SYSTEM_PROMPT = """당신은 기술사업계획서에서 기술 요소를 추출하는 전문가입니다.
다음 JSON 형식으로 반환하세요:
{
  "elements": [
    {
      "name": "기술/컴포넌트명",
      "category": "핵심기술|구현방법|성능지표|입출력|차별점",
      "description": "설명",
      "metrics": ["성능수치1", "성능수치2"],
      "chunk_refs": ["청크ID1"]
    }
  ],
  "differentiators": ["차별화 요소1", "차별화 요소2"],
  "performance_claims": [{"metric": "지표명", "value": "값", "condition": "조건"}]
}"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def run(self, chunks: list[dict], doc_profile: dict) -> dict:
        # 기술 관련 청크 필터링
        tech_chunks = self._filter_tech_chunks(chunks)
        text = "\n\n".join([f"[청크 {c.get('chunk_id','')}]\n{c.get('text','')[:800]}" for c in tech_chunks[:8]])
        
        if self.llm:
            try:
                return await self._call_llm(text, doc_profile)
            except Exception as e:
                print(f"[TechExtraction] LLM 오류, 폴백: {e}")
        
        return self._rule_based(chunks)
    
    async def _call_llm(self, text: str, doc_profile: dict) -> dict:
        context = f"문서 유형: {doc_profile.get('document_type')}\n핵심 기술: {doc_profile.get('main_technology')}"
        return await self.llm.generate_json(
            system=self.SYSTEM_PROMPT,
            user=f"{context}\n\n{text}"
        )
    
    def _filter_tech_chunks(self, chunks: list[dict]) -> list[dict]:
        import re
        tech_keywords = r'기술|알고리즘|시스템|모델|처리|분석|인식|검출|분류|예측|최적화|성능|정확도'
        return [c for c in chunks if re.search(tech_keywords, c.get('text', ''))]
    
    def _rule_based(self, chunks: list[dict]) -> dict:
        import re
        all_text = " ".join([c.get("text", "") for c in chunks])
        metrics = re.findall(r'\d+\.?\d*\s*[%％배ms초]', all_text)
        return {
            "elements": [{"name": "미추출 (LLM 미설정)", "category": "핵심기술", "description": "", "metrics": metrics[:5], "chunk_refs": []}],
            "differentiators": [],
            "performance_claims": [{"metric": m, "value": m, "condition": ""} for m in metrics[:5]]
        }
