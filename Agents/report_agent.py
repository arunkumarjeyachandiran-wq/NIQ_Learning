"""
Report Agent
"""

from research_state import ResearchState
from llm_utils import get_llm, should_use_llm
from .Models import ReportOutput
from prompt_utils import load_prompt


def report_agent(state: ResearchState) -> ResearchState:
    print("\n" + "=" * 60)
    print("REPORT AGENT: Creating report structure")
    print("=" * 60)

    analysis = state.get("analysis")
    review = state.get("review")
    if analysis is None or review is None:
        raise ValueError("Analysis and review outputs are required for report generation.")

    if not should_use_llm():
        state["report"] = ReportOutput(
            title="Mock Research Report",
            executive_summary="This report summarizes the findings and recommendations.",
            sections=[
                {"heading": "Mock Section 1", "content": "Mock content for section 1."},
                {"heading": "Mock Section 2", "content": "Mock content for section 2."},
            ],
            tables=[],
            charts=[],
            images=[],
            conclusion="This is a mock conclusion."
        )
        return state

    llm = get_llm()
    structured_llm = llm.with_structured_output(ReportOutput)

    prompt = load_prompt("report_agent:v1")
    messages = prompt.invoke(
        {   "topic": state["topic"],
            "research_notes": state.get("research_notes").model_dump() if hasattr(state.get("research_notes"), "model_dump") else state.get("research_notes"),
            "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis,
            "review": review.model_dump() if hasattr(review, "model_dump") else review,
        }
    )

    print("\nInvoking report agent with analysis and review")
    report = structured_llm.invoke(messages)

    state["report"] = report
    print("\nReport generated and stored in state.")
    return state
