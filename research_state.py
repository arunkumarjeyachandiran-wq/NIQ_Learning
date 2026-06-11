"""
Shared state definition for the Research Team Agent workflow.
"""

from typing import TypedDict

from Agents.Models import (
    PlannerOutput,
    ResearchOutput,
    AnalysisOutput,
    ReviewOutput,
    ReportOutput,
    AssetsOutput,
)


class ResearchState(TypedDict, total=False):
    topic: str
    plan: PlannerOutput
    research_notes: ResearchOutput
    analysis: AnalysisOutput
    review: ReviewOutput
    report: ReportOutput
    assets: AssetsOutput
    pdf_path: str
