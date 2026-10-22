"""
app/main.py
===========
기술사업계획서 OCR·청킹 및 멀티 에이전트 유사도 판별 솔루션
Streamlit 메인 대시보드
"""

import asyncio
import io
import json
import os
import sys
from pathlib import Path

# 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.config import get_settings
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.agents.orchestrator import AgentOrchestrator
from app.agents.llm_client import GeminiClient
from app.ocr.pipeline import OCRPipeline

# 페이지 기본 설정
st.set_page_config(
    page_title="기술사업계획서 OCR 및 유사도 판별 솔루션",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_app_context():
    """핵심 파이프라인 및 에이전트 인스턴스 초기화 및 캐싱."""
    settings = get_settings()
    pipeline = OCRPipeline(settings)
    vector_store = VectorStore(persist_path=settings.CHROMA_DB_PATH)
    bm25_index = BM25Index(index_path=os.path.join(settings.STORAGE_BASE_PATH, "bm25"))
    embedder = EmbeddingService(
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        model_name=settings.VERTEX_AI_EMBEDDING_MODEL,
    )
    llm_client = GeminiClient(
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        model_name=settings.VERTEX_AI_LLM_MODEL,
    )
    orchestrator = AgentOrchestrator(
        settings=settings,
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedder=embedder,
        llm_client=llm_client if llm_client.is_available else None,
    )
    return {
        "settings": settings,
        "pipeline": pipeline,
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "embedder": embedder,
        "llm_client": llm_client,
        "orchestrator": orchestrator,
    }


def main():
    ctx = get_app_context()
    pipeline: OCRPipeline = ctx["pipeline"]
    orchestrator: AgentOrchestrator = ctx["orchestrator"]
    vector_store: VectorStore = ctx["vector_store"]
    bm25_index: BM25Index = ctx["bm25_index"]
    embedder: EmbeddingService = ctx["embedder"]
    settings = ctx["settings"]

    # 사이드바
    st.sidebar.title("📑 KPC AI 네이티브 솔루션")
    st.sidebar.markdown(
        """
    **기술사업계획서 OCR·청킹 & 멀티 에이전트 유사도 판별**  
    *SPEC 1. OCR·문서 구조화·청킹*  
    *SPEC 2. 멀티 에이전트 유사도 분석*
    """
    )
    st.sidebar.divider()

    # 시스템 상태 표시
    st.sidebar.markdown("### ⚙️ 시스템 구동 상태")
    st.sidebar.write(f"- **임베딩 모델:** `{embedder.model_version}`")
    st.sidebar.write(f"- **LLM 클라이언트:** `{'Gemini 연결됨' if ctx['llm_client'].is_available else '더미/규칙 모드 (로컬)'}`")
    st.sidebar.write(f"- **색인된 청크 수:** `{vector_store.count()}개`")
    st.sidebar.divider()

    menu = st.sidebar.radio(
        "메뉴 선택",
        ["1. 문서 업로드 & OCR 청킹", "2. 등록 문서 보관소", "3. 멀티 에이전트 유사도 분석", "4. 평가 및 감사 로그"],
    )

    # 1. 문서 업로드 & OCR 청킹
    if menu == "1. 문서 업로드 & OCR 청킹":
        st.header("📄 문서 등록 및 OCR·구조화 청킹")
        st.markdown(
            """
        기술사업계획서(PDF 등)를 업로드하면 자동으로 다음 파이프라인이 수행됩니다:
        1. **무결성 검사 및 해시 생성** → 2. **텍스트/스캔 감지 및 적응형 OCR** → 3. **표·제목 레이아웃 분석** → 4. **의미 단위 청킹 & 품질 검사**
        """
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader("PDF 문서 업로드", type=["pdf"])
            tenant_id = st.text_input("테넌트 / 기관 ID", value="default")
            force_reprocess = st.checkbox("중복 검사 건너뛰고 강제 재처리", value=False)

            if uploaded_file is not None:
                if st.button("🚀 OCR 파이프라인 및 청킹 실행", type="primary"):
                    with st.spinner("파이프라인 실행 중... (레이아웃 파싱 및 청킹 수행)"):
                        file_bytes = uploaded_file.read()
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            doc = loop.run_until_complete(
                                pipeline.process(
                                    file_bytes=file_bytes,
                                    filename=uploaded_file.name,
                                    tenant_id=tenant_id,
                                    force_reprocess=force_reprocess,
                                )
                            )

                            # 바로 벡터 DB & BM25에 색인
                            chunks = pipeline.get_chunks(doc.document_id)
                            texts = [c.text for c in chunks]
                            chunk_ids = [c.chunk_id for c in chunks]
                            metadatas = [
                                {
                                    "document_id": c.document_id,
                                    "page_start": c.page_start,
                                    "page_end": c.page_end,
                                    "chunk_type": c.chunk_type.value if hasattr(c.chunk_type, "value") else str(c.chunk_type),
                                    "section_path": " > ".join(c.section_path) if c.section_path else "",
                                }
                                for c in chunks
                            ]

                            embeddings = loop.run_until_complete(embedder.embed_texts(texts))
                            vector_store.upsert_chunks(chunk_ids, embeddings, texts, metadatas)
                            bm25_index.add_chunks(chunk_ids, texts, [doc.document_id] * len(chunk_ids))

                            st.success(f"문서 처리 완료! (문서 ID: {doc.document_id})")
                            st.session_state["last_doc_id"] = doc.document_id
                        except Exception as e:
                            st.error(f"처리 실패: {e}")

        with col2:
            last_id = st.session_state.get("last_doc_id")
            if last_id:
                doc = pipeline.get_status(last_id)
                chunks = pipeline.get_chunks(last_id)
                if doc:
                    st.subheader("📊 처리 요약 및 품질 지표")
                    st.write(f"- **문서 ID:** `{doc.document_id}`")
                    st.write(f"- **상태:** `{doc.status.value}`")
                    st.write(f"- **페이지 수:** `{doc.page_count}p`")
                    st.write(f"- **생성된 청크 수:** `{len(chunks)}개`")
                    st.write(f"- **사용된 OCR 엔진:** `{doc.ocr_engine}` ({doc.model_version})")

                    if doc.quality_summary:
                        st.json(doc.quality_summary)

                    with st.expander("🔍 생성된 청크 미리보기 (Top 5)"):
                        for i, chk in enumerate(chunks[:5]):
                            st.markdown(f"**[{chk.chunk_type}] 청크 #{i+1} (P.{chk.page_start})** - 섹션: `{' > '.join(chk.section_path)}`")
                            st.info(chk.text[:250] + ("..." if len(chk.text) > 250 else ""))

    # 2. 등록 문서 보관소
    elif menu == "2. 등록 문서 보관소":
        st.header("🗄️ 등록된 문서 보관소")
        docs = pipeline.list_documents()

        if not docs:
            st.info("등록된 문서가 없습니다. 1번 메뉴에서 문서를 업로드해 주세요.")
        else:
            table_data = []
            for d in docs:
                table_data.append({
                    "문서 ID": d.document_id,
                    "파일명": d.original_filename,
                    "페이지 수": d.page_count,
                    "상태": d.status.value,
                    "등록 시각": d.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(d.created_at, "strftime") else str(d.created_at),
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            sel_doc_id = st.selectbox("상세 청크 및 메타데이터를 볼 문서 선택", [d.document_id for d in docs])
            if sel_doc_id:
                doc_detail = pipeline.get_status(sel_doc_id)
                chunks_detail = pipeline.get_chunks(sel_doc_id)

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown("#### 메타데이터")
                    st.json(doc_detail.model_dump())

                with col_b:
                    st.markdown(f"#### 청크 목록 ({len(chunks_detail)}개)")
                    for c in chunks_detail:
                        with st.expander(f"청크 #{c.chunk_index} [{c.chunk_type}] P.{c.page_start} | {' > '.join(c.section_path)}"):
                            st.write(c.text)
                            st.caption(f"키워드: {', '.join(c.keywords)} | 해시: {c.chunk_hash}")

    # 3. 멀티 에이전트 유사도 분석
    elif menu == "3. 멀티 에이전트 유사도 분석":
        st.header("🧠 멀티 에이전트 기반 유사도 판별 분석")
        st.markdown(
            """
        7단계 에이전트 오케스트레이션을 통해 다차원 유사도를 평가하고 근거를 도출합니다:
        - **문서 이해** → **기술 요소 추출** → **기술 분류** → **하이브리드 후보 검색** → **6대 차원 유사도 채점** → **근거 검증** → **심사 리포트 생성**
        """
        )

        docs = pipeline.list_documents()
        if len(docs) < 1:
            st.warning("분석을 실행하기 위해 최소 1개 이상의 문서가 등록되어 있어야 합니다.")
            return

        doc_options = {f"{d.original_filename} ({d.document_id})": d.document_id for d in docs}
        selected_label = st.selectbox("분석 대상 신규 사업계획서 선택", list(doc_options.keys()))
        query_doc_id = doc_options[selected_label]

        top_k = st.slider("비교할 상위 후보 수 (Top-K)", min_value=1, max_value=10, value=3)

        if st.button("🔍 멀티 에이전트 유사도 정밀 분석 시작", type="primary"):
            with st.spinner("멀티 에이전트들이 협력하여 분석을 수행 중입니다..."):
                chunks = pipeline.get_chunks(query_doc_id)
                chunk_dicts = [c.model_dump() for c in chunks]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    orchestrator.analyze(
                        query_document_id=query_doc_id,
                        query_chunks=chunk_dicts,
                        top_k=top_k,
                    )
                )

                st.session_state["last_analysis"] = result
                st.success("분석이 완료되었습니다!")

        analysis_res = st.session_state.get("last_analysis")
        if analysis_res and analysis_res.query_document_id == query_doc_id:
            st.divider()
            st.subheader(f"📋 유사도 분석 리포트 (분석 ID: `{analysis_res.analysis_id}`)")

            if not analysis_res.candidates:
                st.info("비교 대상 후보 문서가 없거나 유사 후보가 검색되지 않았습니다.")
            else:
                for cand in analysis_res.candidates:
                    score = cand["overall_similarity_score"]
                    color = "red" if score >= 70 else ("orange" if score >= 40 else "green")

                    st.markdown(
                        f"""
                    ### 🎯 후보 문서: `{cand['document_id']}` (순위 #{cand['rank']})
                    **종합 유사성 지표:** <span style='font-size:24px; color:{color}; font-weight:bold;'>{score} / 100</span>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.info(f"💡 **전문가 검토 권고:** {cand.get('review_recommendation', 'N/A')}")

                    # 6대 차원별 점수 시각화
                    dims = cand.get("dimension_scores", {})
                    dim_df = pd.DataFrame([
                        {"차원": "핵심 기술/해결 원리 (30%)", "점수 (100)": round(dims.get("core_technology", 0) * 100, 1)},
                        {"차원": "문제 정의/적용 분야 (20%)", "점수 (100)": round(dims.get("problem_domain", 0) * 100, 1)},
                        {"차원": "시스템 구조/구현 (15%)", "점수 (100)": round(dims.get("system_architecture", 0) * 100, 1)},
                        {"차원": "성능·검증 지표 (15%)", "점수 (100)": round(dims.get("performance_metrics", 0) * 100, 1)},
                        {"차원": "사업화/시장 전략 (10%)", "점수 (100)": round(dims.get("commercialization", 0) * 100, 1)},
                        {"차원": "키워드/표현 유사도 (10%)", "점수 (100)": round(dims.get("keyword_overlap", 0) * 100, 1)},
                    ])

                    c_left, c_right = st.columns([1, 1])
                    with c_left:
                        st.markdown("**차원별 유사도 점수표**")
                        st.dataframe(dim_df, use_container_width=True)

                    with c_right:
                        st.markdown("**차원별 점수 차트**")
                        st.bar_chart(dim_df.set_index("차원"))

                    # 유사 구간 및 차이점 뷰어
                    with st.expander("📌 유사 근거 구간 및 차이점 상세 확인"):
                        st.markdown("**주요 차이점 및 구별 요소:**")
                        for diff in cand.get("differences", []):
                            st.write(f"- {diff}")

                        st.markdown("**주의 및 품질 플래그:**")
                        for cav in cand.get("quality_and_caveats", []):
                            st.caption(f"⚠️ {cav}")

                        st.markdown("**유사 구간 매칭 정보:**")
                        for passage in cand.get("matched_passages", []):
                            st.json(passage)

                    st.divider()

    # 4. 평가 및 감사 로그
    elif menu == "4. 평가 및 감사 로그":
        st.header("🛡️ 시스템 감사 추적 및 전문가 평가 피드백")

        last_analysis = st.session_state.get("last_analysis")
        if last_analysis:
            st.subheader(f"최근 분석 감사 로그 (`{last_analysis.analysis_id}`)")
            trace_df = pd.DataFrame(last_analysis.audit_trace)
            st.dataframe(trace_df, use_container_width=True)

        st.subheader("✍️ 심사 전문가 피드백 등록")
        with st.form("expert_feedback_form"):
            an_id = st.text_input("분석 ID", value=last_analysis.analysis_id if last_analysis else "")
            cand_id = st.text_input("후보 문서 ID")
            label = st.selectbox(
                "판정 등급",
                ["high_similarity (높은 유사성)", "partial (부분 유사성)", "low (낮은 유사성)", "hold (판단 보류)"],
            )
            comment = st.text_area("심사자 검토 의견 및 근거")
            submitted = st.form_submit_button("피드백 저장")
            if submitted:
                fb_path = Path("./storage/reviews.jsonl")
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fb_path, "a", encoding="utf-8") as f:
                    rec = {
                        "analysis_id": an_id,
                        "candidate_id": cand_id,
                        "decision_label": label,
                        "rationale": comment,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st.success("전문가 심사 피드백이 저장되었습니다.")


if __name__ == "__main__":
    main()
