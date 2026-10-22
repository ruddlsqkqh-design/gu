"""Document Understanding Agent - 섹션·목적·핵심 주장 파악"""
import json

class DocumentUnderstandingAgent:
    """
    구조화 문서와 청크로부터 섹션 요약, 문서 유형, 핵심 주장을 파악.
    LLM 사용 시: Gemini 호출. 미설정 시: 규칙 기반 폴백.
    """
    SYSTEM_PROMPT = """당신은 기술사업계획서 분석 전문가입니다.
주어진 문서 청크들을 분석하여 다음을 JSON으로 반환하세요:
{
  "document_type": "기술사업계획서 | 기술개발계획서 | 기타",
  "main_technology": "핵심 기술명",
  "problem_statement": "해결하려는 문제 요약",
  "solution_overview": "제안하는 솔루션 요약",
  "target_market": "목표 시장/산업",
  "section_summaries": [{"section": "섹션명", "summary": "요약"}],
  "key_claims": ["핵심 주장1", "핵심 주장2"]
}"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def run(self, chunks: list[dict]) -> dict:
        # 대표 청크 선택 (최대 10개, 섹션별 1개)
        selected = self._select_representative_chunks(chunks)
        text_summary = "\n\n".join([f"[{c.get('section_path', '')}]\n{c.get('text', '')[:500]}" for c in selected])
        
        if self.llm:
            try:
                return await self._call_llm(text_summary)
            except Exception as e:
                print(f"[DocUnderstanding] LLM 오류, 폴백: {e}")
        
        return self._rule_based_analysis(chunks)
    
    async def _call_llm(self, text: str) -> dict:
        response = await self.llm.generate_json(
            system=self.SYSTEM_PROMPT,
            user=f"다음 기술사업계획서를 분석하세요:\n\n{text}"
        )
        return response
    
    def _rule_based_analysis(self, chunks: list[dict]) -> dict:
        """LLM 없이 규칙 기반으로 기본 분석"""
        sections = list(set([c.get("section_path", ["미분류"])[0] if c.get("section_path") else "미분류" for c in chunks]))
        all_text = " ".join([c.get("text", "") for c in chunks[:5]])
        
        return {
            "document_type": "기술사업계획서",
            "main_technology": self._extract_keywords(all_text)[:3],
            "problem_statement": "규칙 기반 분석 (LLM 미설정)",
            "solution_overview": "규칙 기반 분석 (LLM 미설정)",
            "target_market": "미추출",
            "section_summaries": [{"section": s, "summary": "미추출"} for s in sections[:5]],
            "key_claims": self._extract_keywords(all_text)[:5]
        }
    
    def _select_representative_chunks(self, chunks: list[dict]) -> list[dict]:
        seen_sections = set()
        selected = []
        for c in chunks:
            section = str(c.get("section_path", []))
            if section not in seen_sections:
                seen_sections.add(section)
                selected.append(c)
            if len(selected) >= 10:
                break
        return selected
    
    def _extract_keywords(self, text: str) -> list[str]:
        import re
        # 한국어 명사구 간단 추출 (2글자 이상 한국어 연속)
        words = re.findall(r'[가-힣]{2,}', text)
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:10]
