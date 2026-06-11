"""
Reviewer Agent Node
"""

from research_state import ResearchState
from llm_utils import get_llm, should_use_llm
from .Models import ReviewOutput
from prompt_utils import load_prompt



def reviewer_agent(state: ResearchState) -> ResearchState:
    """Validate analysis output and detect unsupported claims."""
    print("\n" + "=" * 60)
    print("REVIEWER AGENT: Reviewing analysis")
    print("=" * 60)

    research_notes = state.get("research_notes")
    analysis = state.get("analysis")
    if research_notes is None or analysis is None:
        raise ValueError("Research notes and analysis are required for review.")

    if not should_use_llm():
        state["review"] = ReviewOutput(
            approved=True,
            confidence=92,
            issues=[],
            suggestions=[],
        )
        return state

    llm = get_llm()
    structured_llm = llm.with_structured_output(ReviewOutput)

    prompt = load_prompt("reviewer_agent:v1")
    messages = prompt.invoke(
        {   "topic": state["topic"],
            "research_notes": research_notes.model_dump() if hasattr(research_notes, "model_dump") else research_notes,
            "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis,
        }
    )

    print("\nInvoking reviewer agent with analysis and evidence")
    review = structured_llm.invoke(messages)

    state["review"] = review
    print("\nReview completed and stored in state.")
    return state
