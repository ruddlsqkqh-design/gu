"""멀티 에이전트 유사도 판별 오케스트레이터"""
import uuid, json, asyncio
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

@dataclass
class AnalysisResult:
    analysis_id: str
    query_document_id: str
    status: str  # RUNNING, COMPLETED, FAILED
    candidates: list[dict] = field(default_factory=list)
    policy_version: str = "similarity-policy-0.1"
    embedding_model_version: str = ""
    audit_trace: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    error: str | None = None

class AgentOrchestrator:
    """
    유사도 판별 멀티 에이전트 오케스트레이터.
    
    처리 흐름:
    1. 권한·문서 상태 확인
    2. 신규 문서 이해 (DocumentUnderstandingAgent)
    3. 기술 요소 추출 (TechnologyExtractionAgent)
    4. 기술 분류 (ClassificationAgent)
    5. 후보 검색 (CandidateRetrievalAgent)
    6. 다차원 유사도 계산 (SimilarityScoringAgent)
    7. 근거 검증 (EvidenceValidationAgent)
    8. 리포트 생성 (ReportAgent)
    """
    def __init__(self, settings, vector_store, bm25_index, embedder, llm_client=None):
        self.settings = settings
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.llm_client = llm_client
        
        # 에이전트 초기화
        from app.agents.doc_understanding import DocumentUnderstandingAgent
        from app.agents.tech_extraction import TechnologyExtractionAgent
        from app.agents.classification import ClassificationAgent
        from app.agents.candidate_retrieval import CandidateRetrievalAgent
        from app.agents.similarity_scoring import SimilarityScoringAgent
        from app.agents.evidence_validation import EvidenceValidationAgent
        from app.agents.report_agent import ReportAgent
        
        self.doc_agent = DocumentUnderstandingAgent(llm_client)
        self.tech_agent = TechnologyExtractionAgent(llm_client)
        self.class_agent = ClassificationAgent(llm_client)
        self.retrieval_agent = CandidateRetrievalAgent(vector_store, bm25_index, embedder)
        self.scoring_agent = SimilarityScoringAgent(llm_client)
        self.evidence_agent = EvidenceValidationAgent(llm_client)
        self.report_agent = ReportAgent()
        
        self._results: dict[str, AnalysisResult] = {}
    
    async def analyze(self, query_document_id: str, query_chunks: list[dict],
                      corpus_scope: dict | None = None, top_k: int = 10) -> AnalysisResult:
        """유사도 분석 실행"""
        analysis_id = f"analysis-{uuid.uuid4().hex[:8]}"
        result = AnalysisResult(
            analysis_id=analysis_id,
            query_document_id=query_document_id,
            status="RUNNING",
            embedding_model_version=self.embedder.model_version
        )
        self._results[analysis_id] = result
        
        try:
            self._log(result, "orchestrator", "분석 시작", {"query_doc": query_document_id})
            
            # Step 2: 문서 이해
            doc_profile = await self.doc_agent.run(query_chunks)
            self._log(result, "doc_understanding", "완료", doc_profile)
            
            # Step 3: 기술 요소 추출
            tech_profile = await self.tech_agent.run(query_chunks, doc_profile)
            self._log(result, "tech_extraction", "완료", {"elements": len(tech_profile.get("elements", []))})
            
            # Step 4: 분류
            classification = await self.class_agent.run(tech_profile)
            self._log(result, "classification", "완료", classification)
            
            # Step 5: 후보 검색
            candidates = await self.retrieval_agent.run(
                query_chunks=query_chunks,
                tech_profile=tech_profile,
                corpus_scope=corpus_scope,
                exclude_doc_id=query_document_id,
                top_n=min(50, self.settings.TOP_N_RETRIEVAL)
            )
            self._log(result, "candidate_retrieval", f"후보 {len(candidates)}건 검색", {})
            
            # Step 6: 유사도 계산 (Top-K만)
            top_candidates = candidates[:top_k]
            scored = await self.scoring_agent.run(query_chunks, tech_profile, top_candidates)
            self._log(result, "similarity_scoring", "완료", {"scored": len(scored)})
            
            # Step 7: 근거 검증
            validated = await self.evidence_agent.run(scored)
            self._log(result, "evidence_validation", "완료", {})
            
            # Step 8: 리포트
            report = self.report_agent.run(validated, doc_profile, tech_profile)
            
            result.candidates = report
            result.status = "COMPLETED"
            result.completed_at = datetime.utcnow().isoformat()
            
        except Exception as e:
            result.status = "FAILED"
            result.error = str(e)
            result.completed_at = datetime.utcnow().isoformat()
            self._log(result, "orchestrator", f"오류: {e}", {})
        
        return result
    
    def get_result(self, analysis_id: str) -> AnalysisResult | None:
        return self._results.get(analysis_id)
    
    def _log(self, result: AnalysisResult, agent: str, event: str, data: dict) -> None:
        result.audit_trace.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent,
            "event": event,
            "data": data
        })
