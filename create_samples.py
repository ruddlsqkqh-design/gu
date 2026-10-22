"""
테스트용 샘플 PDF 기술사업계획서 2종 생성 스크립트
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 폰트 등록 (Windows 기본 맑은 고딕 사용 시도)
font_name = "Helvetica"
malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
if malgun_path.exists():
    try:
        pdfmetrics.registerFont(TTFont("Malgun", str(malgun_path)))
        font_name = "Malgun"
    except Exception:
        pass

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Heading1"],
    fontName=font_name,
    fontSize=18,
    leading=22,
    spaceAfter=15,
)
h1_style = ParagraphStyle(
    "H1Style",
    parent=styles["Heading2"],
    fontName=font_name,
    fontSize=14,
    leading=18,
    spaceBefore=10,
    spaceAfter=8,
)
h2_style = ParagraphStyle(
    "H2Style",
    parent=styles["Heading3"],
    fontName=font_name,
    fontSize=12,
    leading=15,
    spaceBefore=8,
    spaceAfter=6,
)
body_style = ParagraphStyle(
    "BodyStyle",
    parent=styles["Normal"],
    fontName=font_name,
    fontSize=10,
    leading=14,
    spaceAfter=6,
)


def create_sample_a(output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = [
        Paragraph("기술사업계획서: 클라우드 네이티브 OCR 및 구조화 엔진", title_style),
        Paragraph("제 1 장 기술 개발의 개요", h1_style),
        Paragraph("1.1 개발 배경 및 필요성", h2_style),
        Paragraph(
            "본 사업은 금융 및 공공기관의 비정형 기술문서로부터 텍스트, 표, 레이아웃을 고정밀로 추출하는 클라우드 네이티브 OCR 시스템을 구축하는 것을 목표로 한다.",
            body_style,
        ),
        Paragraph("1.2 핵심 기술 요소", h2_style),
        Paragraph(
            "Google Document AI 및 Vision API 어댑터 패턴을 결합하여 스캔 문서와 디지털 PDF를 모두 지원하며, 레이아웃 보존 알고리즘을 통해 표와 문단 구조를 원형 그대로 유지한다.",
            body_style,
        ),
        Paragraph("제 2 장 목표 성능 및 지표", h1_style),
    ]

    table_data = [
        [Paragraph("항목", body_style), Paragraph("목표치", body_style), Paragraph("달성방안", body_style)],
        [Paragraph("문자인식 정확도", body_style), Paragraph("98.5%", body_style), Paragraph("어댑터 앙상블", body_style)],
        [Paragraph("표 구조 복원율", body_style), Paragraph("95.0%", body_style), Paragraph("셀 분할 알고리즘", body_style)],
        [Paragraph("응답 시간", body_style), Paragraph("3.0초 이하", body_style), Paragraph("비동기 처리 파이프라인", body_style)],
    ]
    t = Table(table_data, colWidths=[120, 100, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    doc.build(story)


def create_sample_b(output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = [
        Paragraph("기술사업계획서: 멀티 에이전트 기반 사업계획서 유사도 판별 솔루션", title_style),
        Paragraph("제 1 장 과제 개요", h1_style),
        Paragraph("1.1 과제 목표 및 해결 과제", h2_style),
        Paragraph(
            "신규 제출된 기술사업계획서를 기존 지식 베이스와 비교하여 기술적 중복성 및 유사도를 판별하고, 심사위원에게 객관적인 근거 리포트를 제공하는 솔루션을 개발한다.",
            body_style,
        ),
        Paragraph("1.2 멀티 에이전트 및 하이브리드 검색 아키텍처", h2_style),
        Paragraph(
            "문서 이해, 기술 추출, 분류, 후보 검색, 다차원 채점, 근거 검증 등 7종의 특화 에이전트가 협업하며, ChromaDB 벡터 검색과 BM25 키워드 검색의 RRF 결합을 수행한다.",
            body_style,
        ),
        Paragraph("제 2 장 세부 평가 차원 및 목표 지표", h1_style),
    ]

    table_data = [
        [Paragraph("평가 차원", body_style), Paragraph("가중치", body_style), Paragraph("검증 방식", body_style)],
        [Paragraph("핵심 기술 및 해결원리", body_style), Paragraph("30%", body_style), Paragraph("의미 벡터 유사도", body_style)],
        [Paragraph("문제 정의 및 적용 분야", body_style), Paragraph("20%", body_style), Paragraph("키워드 매칭 및 LLM", body_style)],
        [Paragraph("시스템 구현 및 구조", body_style), Paragraph("15%", body_style), Paragraph("아키텍처 요소 비교", body_style)],
    ]
    t = Table(table_data, colWidths=[140, 80, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.beige),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    doc.build(story)


if __name__ == "__main__":
    out_dir = Path("./storage/sample_docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_sample_a(str(out_dir / "plan_A_ocr_engine.pdf"))
    create_sample_b(str(out_dir / "plan_B_similarity_agent.pdf"))
    print("Sample PDFs created successfully in ./storage/sample_docs")
