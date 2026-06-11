"""
Research Agent Node
"""

from langgraph.prebuilt import create_react_agent
from research_state import ResearchState
from llm_utils import get_llm, should_use_llm, get_mock_research_output
from .tools import web_search
from .Models import ResearchOutput
from prompt_utils import load_prompt


# Create LLM and agent once per module.
llm = get_llm()
agent = create_react_agent(model=llm, tools=[web_search])


def research_agent(state: ResearchState) -> ResearchState:
    """Execute evidence gathering from the investigation plan."""
    print("\n" + "=" * 60)
    print("RESEARCH AGENT: Gathering evidence")
    print("=" * 60)

    plan = state.get("plan")
    if plan is None:
        raise ValueError("Research plan is required before research can begin.")

    if not should_use_llm():
        state["research_notes"] = ResearchOutput(**get_mock_research_output(state["topic"]))
        return state

    prompt = load_prompt("research_agent:v1")
    prompt_value = prompt.invoke(
        {
            "topic": state["topic"],
            "plan": plan.model_dump(),
        }
    )
    messages = prompt_value.to_messages()

    print("\nInvoking research agent with plan and tools")
    result = agent.invoke({"messages": messages})

    research_text = result["messages"][-1].content

    structured_llm = llm.with_structured_output(ResearchOutput)
    research_notes = structured_llm.invoke(
        f"""
Convert the following research notes into the required schema:

{research_text}
"""
    )

    state["research_notes"] = research_notes
    print("\nResearch notes generated and stored in state.")
    return state
