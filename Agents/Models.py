from pydantic import BaseModel, Field
from typing import List, Dict


class PlannerOutput(BaseModel):
    entities: List[str] = Field(default_factory=list)
    research_questions: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    scope: str = ""


class ResearchOutput(BaseModel):
    entities: Dict[str, str] = Field(default_factory=dict)
    facts: List[str] = Field(default_factory=list)
    statistics: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    key_insights: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    tradeoffs: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    approved: bool = Field(description="Whether the analysis has been validated")
    confidence: int = Field(description="Confidence score from 0 to 100")
    issues: List[str] = Field(default_factory=list, description="Detected issues or unsupported claims")
    suggestions: List[str] = Field(default_factory=list, description="Reviewer suggestions for improvements")


class ReportSection(BaseModel):
    heading: str
    content: str


class TableSpec(BaseModel):
    title: str
    headers: List[str]
    rows: List[List[str]]


class ChartSpec(BaseModel):
    title: str
    chart_type: str = Field(description="chart type, e.g. bar, line, pie, scatter")
    x_label: str
    y_label: str
    labels: List[str]
    values: List[float]


class ImageSpec(BaseModel):
    title: str
    search_query: str
    reason: str


class ReportOutput(BaseModel):
    title: str
    executive_summary: str
    sections: List[ReportSection] = Field(default_factory=list)
    conclusion: str
    tables: List[TableSpec] = Field(default_factory=list)
    charts: List[ChartSpec] = Field(default_factory=list)
    images: List[ImageSpec] = Field(default_factory=list)


class AssetsOutput(BaseModel):
    tables: List[TableSpec] = Field(default_factory=list)
    charts: List[ChartSpec] = Field(default_factory=list)
    images: List[ImageSpec] = Field(default_factory=list)


