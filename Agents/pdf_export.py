"""PDF Export Agent Node"""

from research_state import ResearchState
from services.asset_service import build_assets
from services.pdf_service import export_report_pdf


def pdf_export_agent(state: ResearchState) -> ResearchState:
    """Generate assets and export the final report as PDF."""
    print("\n" + "=" * 60)
    print("PDF EXPORT AGENT: Generating assets and exporting PDF")
    print("=" * 60)

    report = state.get("report")
    if report is None:
        raise ValueError("Report content is required before PDF export.")

    assets = build_assets(
        tables=getattr(report, "tables", []),
        charts=getattr(report, "charts", []),
        images=getattr(report, "images", []),
    )

    state["assets"] = assets
    state["pdf_path"] = export_report_pdf(state["topic"], report)

    print(f"\nPDF generated: {state['pdf_path']}")
    return state
