from functools import lru_cache
from langsmith import Client


@lru_cache(maxsize=None)
def load_prompt(prompt_name: str):
    """Load a prompt from LangSmith by name."""
    client = Client()
    return client.pull_prompt(prompt_name)
