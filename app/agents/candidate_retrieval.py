"""Candidate Retrieval Agent - 하이브리드 검색"""
import asyncio

class CandidateRetrievalAgent:
    """
    벡터 검색 + BM25 키워드 검색 하이브리드.
    Reciprocal Rank Fusion (RRF)으로 점수 융합.
    """
    RRF_K = 60  # RRF 상수
    
    def __init__(self, vector_store, bm25_index, embedder):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
    
    async def run(self, query_chunks: list[dict], tech_profile: dict,
                  corpus_scope: dict | None = None, exclude_doc_id: str = "",
                  top_n: int = 50) -> list[dict]:
        """
        하이브리드 검색 후 RRF 융합.
        반환: [{"document_id", "rrf_score", "vector_hits": [...], "bm25_hits": [...]}]
        """
        # 쿼리 텍스트 구성
        query_text = self._build_query_text(query_chunks, tech_profile)
        
        # 벡터 검색
        query_emb = (await self.embedder.embed_texts([query_text]))[0]
        where_filter = None
        if corpus_scope and corpus_scope.get("tenant_id"):
            where_filter = {"tenant_id": corpus_scope["tenant_id"]}
        
        vector_hits = self.vector_store.search(query_emb, n_results=top_n, where=where_filter)
        bm25_hits = self.bm25_index.search(query_text, n_results=top_n)
        
        # 제외 문서 필터링
        vector_hits = [h for h in vector_hits if h["document_id"] != exclude_doc_id]
        bm25_hits = [h for h in bm25_hits if h["document_id"] != exclude_doc_id]
        
        # RRF 융합
        return self._rrf_fusion(vector_hits, bm25_hits, top_n)
    
    def _build_query_text(self, chunks: list[dict], tech_profile: dict) -> str:
        chunk_texts = " ".join([c.get("text", "")[:200] for c in chunks[:5]])
        tech_text = " ".join([e.get("name", "") + " " + e.get("description", "")
                              for e in tech_profile.get("elements", [])[:3]])
        return f"{tech_text} {chunk_texts}".strip()
    
    def _rrf_fusion(self, vector_hits: list[dict], bm25_hits: list[dict], top_n: int) -> list[dict]:
        """Reciprocal Rank Fusion"""
        doc_scores: dict[str, dict] = {}
        
        for rank, hit in enumerate(vector_hits):
            doc_id = hit["document_id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"document_id": doc_id, "rrf_score": 0.0, "vector_hits": [], "bm25_hits": []}
            doc_scores[doc_id]["rrf_score"] += 1.0 / (self.RRF_K + rank + 1)
            doc_scores[doc_id]["vector_hits"].append(hit)
        
        for rank, hit in enumerate(bm25_hits):
            doc_id = hit["document_id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"document_id": doc_id, "rrf_score": 0.0, "vector_hits": [], "bm25_hits": []}
            doc_scores[doc_id]["rrf_score"] += 1.0 / (self.RRF_K + rank + 1)
            doc_scores[doc_id]["bm25_hits"].append(hit)
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_docs[:top_n]
