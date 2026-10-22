# 기술사업계획서 OCR·청킹 및 멀티 에이전트 유사도 판별 솔루션

본 솔루션은 기술명세서(**SPEC 1. OCR·문서 구조화·청킹**, **SPEC 2. 멀티 에이전트 기반 유사도 판별**)를 충실히 반영하여 구현된 엔드투엔드 파이프라인 및 웹 대시보드 애플리케이션입니다.

---

## 🏛️ 전체 시스템 아키텍처

```
[PDF / 스캔 문서 업로드]
         │
         ▼
[1. OCR 파이프라인]
  ├─ 텍스트 레이어 검출 (PDFParserAdapter: pdfplumber)
  ├─ 스캔/이미지 어댑터 (Document AI, Vision OCR)
  ├─ 텍스트 정규화 (유니코드 NFC, 전각→반각, 하이픈 연결)
  ├─ 레이아웃 & 섹션 트리 분석 (장/절/항목 계층 구조 복원)
  ├─ 구조 기반 청킹 (SPEC 1 §5 정책: 500~900 토큰, 표 헤더 보존)
  └─ 품질 검사기 (빈 페이지, 낮은 신뢰도, 깨진 문자 탐지)
         │
         ▼
[2. 인덱싱 & 하이브리드 검색]
  ├─ 벡터 임베딩 (Vertex AI text-multilingual-embedding-002 / 고성능 코사인 폴백)
  ├─ 키워드 인덱스 (BM25Okapi)
  └─ RRF (Reciprocal Rank Fusion) 점수 융합
         │
         ▼
[3. 멀티 에이전트 유사도 판별 시스템]
  ├─ Document Understanding Agent : 섹션·문서 목적·핵심 주장 파악
  ├─ Technology Extraction Agent  : 기술 요소·구성·성능 지표 추출
  ├─ Classification Agent         : 기술 분류 체계 태깅
  ├─ Candidate Retrieval Agent    : 하이브리드 후보 문서 Top-N 검색
  ├─ Similarity Scoring Agent     : 6대 차원별 가중 유사도 산출
  ├─ Evidence Validation Agent    : 근거 인용 및 차이점 검증
  └─ Report Agent                 : 심사위원용 종합 리포트 생성
         │
         ▼
[4. 사용자 인터페이스 (Streamlit Web App)]
  ├─ 문서 업로드 & 실시간 청킹·품질 뷰어
  ├─ 등록 문서 보관소 및 청크 상세 탐색
  ├─ 멀티 에이전트 유사도 분석 및 6대 차원 시각화 차트
  └─ 시스템 감사 로그 및 전문가 심사 피드백 기록
```

---

## 🚀 빠른 시작 가이드

### 1. 가상환경 및 패키지
필요한 패키지들이 이미 설치되어 있습니다. (pydantic-settings, pdfplumber, rank-bm25, reportlab, streamlit 등)

### 2. Streamlit 웹 대시보드 실행
```powershell
streamlit run app/main.py
```
브라우저에서 `http://localhost:8501`로 접속하여 문서 업로드, 청킹 확인 및 유사도 분석을 즉시 수행할 수 있습니다.

### 3. REST API 서버 실행 (선택사항)
FastAPI REST API(SPEC 2 §7 엔드포인트 규격 준수):
```powershell
uvicorn app.api.routes:router --reload --port 8000
```

### 4. 단위 및 E2E 테스트 실행
```powershell
# 핵심 단위 테스트 (5종)
python -m unittest tests/test_pipeline.py

# 실제 PDF 생성 및 전체 파이프라인 E2E 테스트
python test_e2e.py
```

---

## 📂 프로젝트 주요 구성 파일

- `app/ocr/` : OCR 어댑터, 정규화, 레이아웃 분석, 청킹 엔진, 품질 검사기
- `app/indexing/` : 고성능 벡터 저장소(NumPy Cosine) 및 BM25 키워드 인덱스
- `app/agents/` : 7종 멀티 에이전트 오케스트레이션 시스템
- `app/models/` : DocumentModel 및 ChunkModel 도메인 스키마
- `app/main.py` : Streamlit 기반 통합 웹 대시보드
- `storage/` : 원본 파일, OCR 결과 JSON, 청크 메타데이터, 벡터 저장소 영속 디렉토리
