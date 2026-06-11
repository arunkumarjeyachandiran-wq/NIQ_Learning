"""
Analysis Agent Node
"""

from research_state import ResearchState
from llm_utils import get_llm, should_use_llm, get_mock_analysis_output
from .Models import AnalysisOutput
from prompt_utils import load_prompt


llm = get_llm()


def analysis_agent(state: ResearchState) -> ResearchState:
    """Compare evidence and generate insights from research notes."""
    print("\n" + "=" * 60)
    print("ANALYSIS AGENT: Analyzing findings")
    print("=" * 60)

    research_notes = state.get("research_notes")
    if research_notes is None:
        raise ValueError("Research notes are required for analysis.")

    if not should_use_llm():
        state["analysis"] = AnalysisOutput(**get_mock_analysis_output(state["topic"]))
        return state

    prompt = load_prompt("analysis_agent:v1")
    structured_llm = llm.with_structured_output(AnalysisOutput)

    messages = prompt.invoke(
        {   "topic": state["topic"],
            "research_notes": research_notes.model_dump() if hasattr(research_notes, "model_dump") else research_notes
        }
    )

    print("\nInvoking analysis agent with research notes")
    analysis = structured_llm.invoke(messages)

    state["analysis"] = analysis
    print("\nAnalysis generated and stored in state.")
    return state
