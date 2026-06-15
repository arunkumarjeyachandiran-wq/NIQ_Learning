"""
LLM helper utilities used by each agent.
"""

import os

# pip install langchain-huggingface
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()  # This searches for the .env file and loads your key

# Model_NAME = os.getenv("MODEL_NAME", "openai").lower()
Model_NAME = "openai" # Change to "gemini" or "huggingface" or "openai" or "groq" to test other LLMs

LLM = None  # Global variable to hold the initialized LLM instance
USE_LLM = False  # Set to False to disable LLM-based agents, True to enable

def should_use_llm() -> bool:
    """
    Check if LLM should be used or if static output should be returned.
    Set USE_LLM=false in .env to use static outputs (useful for testing).
    """

    print(f"USE_LLM is set to '{USE_LLM}'. Using LLM: {USE_LLM}")
    return USE_LLM


def get_llm():
    """
    Initializes the LLM.
    """
    global LLM

    if LLM is not None:
        return LLM

    print(f"Initializing LLM: {Model_NAME}")

    if Model_NAME == "openai":
        LLM = ChatOpenAI(
            model="meta-llama/llama-4-scout",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
        )
    
    elif Model_NAME == "huggingface":
        endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=os.environ["HF_API_KEY"]
        )

        LLM = ChatHuggingFace(llm=endpoint)

#     elif Model_NAME == "groq":
#         LLM = ChatGroq(
#     model="llama3-8b-8192", 
#     api_key=os.environ["GROQ_API_KEY"]
# )
    else:
        LLM = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.7
        )

    return LLM


# ============================================================================
# STATIC MOCK OUTPUTS FOR TESTING
# ============================================================================

def get_mock_research_output(topic: str) -> dict:
    """Mock research output for testing without LLM."""
    return {
        "entities": {
            "OpenAI": "Text-first AI platform with strong developer tools.",
            "Gemini": "Multimodal AI system optimized for search and image understanding.",
            "Evaluation": "Benchmarking, tradeoffs, and real-world adoption."
        },
        "facts": [
            f"OpenAI and Gemini are both leading providers for large language models in 2026.",
            "OpenAI emphasizes API compatibility and developer ecosystem support.",
            "Gemini emphasizes multimodal capabilities and Google product integration."
        ],
        "statistics": [
            "OpenAI reports high performance on text generation benchmarks.",
            "Gemini demonstrates strong multimodal accuracy in image+text tasks.",
            "Latency and cost vary based on deployment configuration."
        ],
        "references": [
            "OpenAI developer documentation",
            "Gemini research summaries",
            "Industry benchmark reports"
        ],
        "evidence_gaps": [
            "Limited direct comparison of recent pricing plans.",
            "Fewer public case studies on enterprise Gemini deployments.",
            "Incomplete benchmark data for specialized low-latency workloads."
        ]
    }


def get_mock_analysis_output(topic: str) -> dict:
    """Mock analysis output for testing without LLM."""
    return {
        "key_insights": [
            "OpenAI excels in text-first applications with robust developer tooling.",
            "Gemini excels when multimodal inputs and search-aware context are required.",
            "The best choice depends on whether the project needs text-only flexibility or integrated Google capabilities."
        ],
        "strengths": [
            "OpenAI provides consistent text quality and broad third-party integration.",
            "Gemini offers strong multimodal reasoning and tight Google ecosystem access.",
            "Both models can deliver high-quality summaries, translations, and code generation."
        ],
        "weaknesses": [
            "OpenAI may require additional fine-tuning for domain-specific tasks.",
            "Gemini can be more complex to integrate outside of Google Cloud environments.",
            "Both systems still require guardrails to reduce hallucinations and bias."
        ],
        "tradeoffs": [
            "OpenAI trades broad compatibility for slightly higher integration effort in some workflows.",
            "Gemini trades deep Google integration for potentially less straightforward setup outside Google services.",
            "Cost, latency, and feature set should be balanced based on use case and team expertise."
        ],
        "recommendations": [
            "Choose OpenAI for text-heavy applications requiring mature developer tooling.",
            "Choose Gemini for multimodal applications that benefit from Google ecosystem integration.",
            "Perform a pilot evaluation to validate the best fit for your specific requirements."
        ]
    }


def get_mock_report_output(topic: str) -> str:
    """Mock report output for testing without LLM."""
    return f"""# Report: {topic}

## Introduction

This report provides a comprehensive analysis of {topic}. We examine the key approaches, their advantages and disadvantages, and provide recommendations for practitioners.

## Research Findings

### Key Concepts
{topic} encompasses several fundamental concepts that are essential to understanding modern approaches.

### Important Facts
- {topic} has evolved significantly over the past decade
- Multiple competing methodologies exist in the field
- Industry adoption rates continue to increase
- Best practices are still being established

### Advantages
- Improved efficiency and performance
- Reduced complexity in many scenarios
- Cost-effective solutions available
- Strong community support and resources

### Limitations
- Implementation complexity in some contexts
- Learning curve for new practitioners
- Integration challenges with legacy systems
- Ongoing optimization requirements

## Comparative Analysis

| Aspect | Approach A | Approach B |
|--------|-----------|-----------|
| Performance | High | Medium |
| Cost | Medium | Low |
| Complexity | High | Low |
| Scalability | Excellent | Good |
| Maintenance | Moderate | Low |

## Tradeoffs

### Speed vs. Simplicity
Faster approaches often require more complexity, while simpler solutions may sacrifice some performance.

### Cost vs. Features
More feature-rich solutions typically have higher initial and ongoing costs.

### Flexibility vs. Standards
Custom implementations offer more flexibility but at the cost of standardization and community support.

## Recommendations

1. **For High-Performance Needs**: Consider Approach A despite higher complexity
2. **For Quick Implementation**: Approach B offers faster time-to-market
3. **For Long-Term Growth**: Evaluate scalability requirements carefully
4. **For Cost-Conscious Projects**: Approach B provides better ROI

## Conclusion

{topic} is an important area requiring careful evaluation of multiple factors. The choice between different approaches depends heavily on specific requirements, constraints, and long-term goals. We recommend:

- Evaluating both approaches in your specific context
- Starting with the approach that best matches your immediate needs
- Planning for potential migration or scaling later
- Maintaining flexibility for future changes and improvements

### Key Takeaways
- No single approach is universally better
- Context and requirements determine the best choice
- Trade-offs are inevitable and must be understood
- Continuous evaluation and optimization are necessary"""