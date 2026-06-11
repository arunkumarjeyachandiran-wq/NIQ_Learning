"""Planner Agent Node"""

from research_state import ResearchState
from llm_utils import get_llm, should_use_llm
from .Models import PlannerOutput
from prompt_utils import load_prompt


def planner_agent(state: ResearchState) -> ResearchState:
    """Generate a structured research plan from the topic."""
    print("\n" + "=" * 60)
    print("PLANNER AGENT: Building investigation plan")
    print("=" * 60)

    if not should_use_llm():
        state["plan"] = PlannerOutput(
            entities=["OpenAI", "Gemini", "LLM evaluation"],
            research_questions=[
                "What are the latest strengths and weaknesses of each model?",
                "What quantitative benchmarks are available?",
                "Which real-world use cases favor each approach?"
            ],
            search_queries=[
                "OpenAI vs Gemini model evaluation 2026",
                "Gemini multimodal strengths and weaknesses",
                "OpenAI developer tooling advantages"
            ],
            scope="Compare capabilities, tradeoffs, and enterprise adoption for the topic."
        )
        return state

    llm = get_llm()
    structured_llm = llm.with_structured_output(PlannerOutput)

    prompt = load_prompt("planner_agent:latest")
    messages = prompt.invoke(
        {
            "topic": state["topic"],
        }
    )

    plan = structured_llm.invoke(messages)
    state["plan"] = plan
    # print(plan)
    print("\nPlanner output stored in state.")
    # raise
    return state
