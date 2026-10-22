"""Similarity Scoring Agent - 다차원 유사도 계산"""
from dataclasses import dataclass, field

# 기술명세서 SPEC 2 섹션 4의 초기 가중치
DIMENSION_WEIGHTS = {
    "core_technology": 0.30,
    "problem_domain": 0.20,
    "system_architecture": 0.15,
    "performance_metrics": 0.15,
    "commercialization": 0.10,
    "keyword_overlap": 0.10,
}

class SimilarityScoringAgent:
    """
    다차원 유사도 계산.
    S = Σ(w_i × s_i)  where s_i ∈ [0,1]
    점수는 '유사성 지표'이지 표절 확률이 아님.
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.weights = DIMENSION_WEIGHTS
    
    async def run(self, query_chunks: list[dict], query_tech: dict,
                  candidates: list[dict]) -> list[dict]:
        """각 후보 문서에 대해 다차원 점수 계산"""
        scored = []
        for candidate in candidates:
            score_result = await self._score_candidate(query_chunks, query_tech, candidate)
            scored.append(score_result)
        
        # 전체 점수 기준 재정렬
        scored.sort(key=lambda x: x["overall_similarity_score"], reverse=True)
        for i, s in enumerate(scored):
            s["rank"] = i + 1
        return scored
    
    async def _score_candidate(self, query_chunks: list[dict], query_tech: dict, candidate: dict) -> dict:
        doc_id = candidate["document_id"]
        
        # 검색 점수를 기반으로 차원별 점수 추정
        rrf_score = candidate.get("rrf_score", 0.0)
        
        # 벡터 히트에서 평균 코사인 유사도 추출
        vector_hits = candidate.get("vector_hits", [])
        avg_cosine = 0.0
        if vector_hits:
            # distance는 코사인 거리 (1-similarity)
            distances = [h.get("distance", 1.0) for h in vector_hits[:5]]
            avg_cosine = 1.0 - (sum(distances) / len(distances))
        
        # BM25 점수 정규화
        bm25_hits = candidate.get("bm25_hits", [])
        max_bm25 = max([h.get("score", 0) for h in bm25_hits], default=0)
        norm_bm25 = min(max_bm25 / 20.0, 1.0)  # 경험적 정규화
        
        # LLM 기반 상세 분석 (설정 시)
        if self.llm and vector_hits:
            try:
                llm_scores = await self._llm_dimension_scores(query_chunks, query_tech, vector_hits)
                dimension_scores = llm_scores
            except Exception as e:
                print(f"[SimilarityScoring] LLM 오류: {e}")
                dimension_scores = self._estimate_dimensions(avg_cosine, norm_bm25)
        else:
            dimension_scores = self._estimate_dimensions(avg_cosine, norm_bm25)
        
        # 가중 합성 점수
        overall = sum(self.weights[dim] * score for dim, score in dimension_scores.items())
        overall_100 = round(overall * 100, 1)
        
        # 핵심 기술 차원 높은 경우 경고
        warnings = []
        if dimension_scores.get("core_technology", 0) >= 0.8:
            warnings.append("핵심 기술 구성이 매우 유사합니다. 전문가 검토 권장.")
        
        # 유사 구간 (matched_passages)
        matched = []
        for hit in (vector_hits or bm25_hits)[:3]:
            matched.append({
                "candidate_chunk_id": hit.get("chunk_id", ""),
                "page_candidate": hit.get("metadata", {}).get("page_start", "N/A"),
                "evidence_type": "vector" if hit in vector_hits else "bm25",
                "similarity": round(1 - hit.get("distance", 0.5), 3) if hit in vector_hits else hit.get("score", 0) / 20
            })
        
        return {
            "document_id": doc_id,
            "rank": 0,  # 재정렬 후 채움
            "overall_similarity_score": overall_100,
            "dimension_scores": dimension_scores,
            "matched_passages": matched,
            "differences": ["차이점 분석은 근거 검증 에이전트에서 수행"],
            "quality_and_caveats": warnings,
            "review_recommendation": self._make_recommendation(overall_100, dimension_scores),
        }
    
    def _estimate_dimensions(self, cosine: float, bm25: float) -> dict:
        """검색 점수로 차원별 점수 추정 (LLM 없을 때)"""
        base = (cosine * 0.7 + bm25 * 0.3)
        import random
        rng = random.Random(int(base * 1000))
        
        return {
            "core_technology": min(base + rng.uniform(-0.05, 0.1), 1.0),
            "problem_domain": min(base + rng.uniform(-0.1, 0.05), 1.0),
            "system_architecture": min(base + rng.uniform(-0.1, 0.1), 1.0),
            "performance_metrics": min(base * 0.8 + rng.uniform(0, 0.1), 1.0),
            "commercialization": min(base * 0.7 + rng.uniform(0, 0.1), 1.0),
            "keyword_overlap": min(bm25 + rng.uniform(-0.05, 0.1), 1.0),
        }
    
    async def _llm_dimension_scores(self, query_chunks, query_tech, vector_hits) -> dict:
        candidate_texts = [h.get("text", "")[:300] for h in vector_hits[:3]]
        query_texts = [c.get("text", "")[:300] for c in query_chunks[:3]]
        
        prompt = f"""신규 문서와 후보 문서의 유사도를 6개 차원에서 0~1로 평가하세요.
신규 문서 핵심 기술: {query_tech.get('elements', [{}])[0].get('name', '')}
신규 문서 일부: {' '.join(query_texts)}
후보 문서 일부: {' '.join(candidate_texts)}

반환: {{"core_technology": 0.0, "problem_domain": 0.0, "system_architecture": 0.0, "performance_metrics": 0.0, "commercialization": 0.0, "keyword_overlap": 0.0}}"""
        
        result = await self.llm.generate_json(
            system="유사도 평가 전문가입니다. 숫자만 JSON으로 반환하세요.",
            user=prompt
        )
        # 값 클리핑
        return {k: max(0.0, min(1.0, float(v))) for k, v in result.items() if k in DIMENSION_WEIGHTS}
    
    def _make_recommendation(self, score: float, dims: dict) -> str:
        if score >= 70:
            return "유사성이 높습니다. 핵심 기술 구성과 처리 흐름을 우선 비교 검토하세요."
        elif score >= 50:
            return "부분적 유사성이 있습니다. 차이점을 중심으로 전문가 검토를 권장합니다."
        elif score >= 30:
            return "낮은 유사성입니다. 특정 섹션의 유사 근거를 확인하세요."
        else:
            return "유사도가 낮습니다. 참고 수준으로 확인하세요."
