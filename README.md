# Research Team Agent - Setup & Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install langchain langchain-openai langgraph python-dotenv
```

### 2. Set Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

Or set it directly:
```bash
export OPENAI_API_KEY=your-api-key-here  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here     # Windows
```

### 3. Run the Workflow
```bash
python research_workflow.py
```

## File Structure
```
agents_tut/
├── main.py                    # Entry point (customize as needed)
├── research_workflow.py       # V1: Basic workflow with 4 agents
├── Workflow.txt              # Architecture documentation
└── .env                      # Environment variables (add this)
```

## What's Implemented (V1)

✅ **Supervisor** - Receives task, initializes workflow  
✅ **Research Agent** - Gathers information (LLM only)  
✅ **Analysis Agent** - Analyzes findings  
✅ **Writer Agent** - Creates final report  
✅ **State Management** - Shared ResearchState through workflow  
✅ **LangGraph Integration** - Linear workflow with nodes and edges  

## What's NOT Yet (by design)

❌ Tools (search, web, documents)  
❌ Reviewer Agent & approval loop  
❌ Multi-agent communication via messages  
❌ A2A (Agent-to-Agent)  
❌ MCP (Model Context Protocol)  

## Next Steps

### To test the current implementation:
1. Set your OpenAI API key
2. Run: `python research_workflow.py`
3. Verify each agent's output

### To upgrade to V2 (add tools):
- Add search tool to Research Agent
- Integrate web_urls tool
- Update research_agent() prompt to use tools

### To upgrade to V3 (add reviewer):
- Add reviewer_agent() node
- Add conditional routing based on approval
- Add max revision counter

## Customization Points

**Change the LLM:**
```python
def get_llm():
    return ChatAnthropic(model="claude-3-sonnet")  # or any other provider
```

**Change the topic:**
```python
topic = "Your research question here"
final_report = run_research_workflow(topic)
```

**Modify agent prompts:**
Edit the `prompt` variable in each agent function.

**Add new state fields:**
```python
class ResearchState(TypedDict):
    topic: str
    research_notes: str
    analysis: str
    report: str
    review: str              # Add new field
    approved: bool           # Add new field
```

## Debugging

Add logging to see what's happening:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check LangGraph visualization:
```python
from IPython.display import display, Image
display(Image(app.get_graph().draw_mermaid_png()))
```

## Common Issues

**ImportError: No module named 'langgraph'**
```bash
pip install langgraph
```

**OpenAI API Key error**
Ensure `OPENAI_API_KEY` is set in environment variables.

**Rate limit errors**
Add delays between API calls or use a smaller model.

## Learning Path

1. **Now**: Understanding basic workflow structure
2. **Next**: Adding tools to Research Agent
3. **Then**: Adding Reviewer Agent with loops
4. **Later**: Converting to distributed services for A2A
