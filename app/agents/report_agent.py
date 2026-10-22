"""Report Agent - 심사자용 리포트 생성"""
from datetime import datetime

class ReportAgent:
    def run(self, validated_candidates: list[dict], 
            doc_profile: dict, tech_profile: dict) -> list[dict]:
        """최종 리포트 데이터 구성"""
        report = []
        for c in validated_candidates:
            report.append({
                "document_id": c["document_id"],
                "rank": c["rank"],
                "overall_similarity_score": c["overall_similarity_score"],
                "dimension_scores": c.get("dimension_scores", {}),
                "matched_passages": c.get("matched_passages", []),
                "differences": c.get("differences", []),
                "quality_and_caveats": c.get("quality_and_caveats", []),
                "review_recommendation": c.get("review_recommendation", ""),
            })
        return report
