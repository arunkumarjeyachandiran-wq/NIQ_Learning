"""
Research Team Agent - LangGraph Workflow
Target architecture: planner → research → analysis → reviewer → report → pdf export.
"""

from langgraph.graph import StateGraph, START, END

from research_state import ResearchState
from Agents.planner_agent import planner_agent
from Agents.research_agent import research_agent
from Agents.analysis_agent import analysis_agent
from Agents.reviewer_agent import reviewer_agent
from Agents.report_agent import report_agent
from Agents.pdf_export import pdf_export_agent

# ============================================================================
# BUILD WORKFLOW
# ============================================================================


def build_workflow():
    """Construct the LangGraph workflow."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("review", reviewer_agent)
    workflow.add_node("report", report_agent)
    workflow.add_node("pdf_export", pdf_export_agent)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "review")
    workflow.add_edge("review", "report")
    workflow.add_edge("report", "pdf_export")
    workflow.add_edge("pdf_export", END)

    return workflow.compile()


# ============================================================================
# EXECUTION
# ============================================================================


def run_research_workflow(topic: str):
    """Execute the research workflow."""
    initial_state = ResearchState(topic=topic, pdf_path="")

    app = build_workflow()
    final_state = app.invoke(initial_state)

    return final_state


# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == "__main__":
#     # Example research topic
#     topic = "Compare YOLO v8 vs DETR for object detection"

#     print("\n" + "="*60)
#     print("RESEARCH WORKFLOW - V1 (No Tools)")
#     print("="*60)
#     print(f"Research Topic: {topic}\n")

#     # Run the workflow
#     final_report = run_research_workflow(topic)

#     # Display result
#     print("\n" + "="*60)
#     print("FINAL REPORT")
#     print("="*60)
#     print(final_report)

#     print("\n" + "="*60)
#     print("WORKFLOW COMPLETE")
#     print("="*60)
