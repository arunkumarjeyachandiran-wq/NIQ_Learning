from langchain_core.tools import tool
from pydantic import BaseModel, Field

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

from ddgs import DDGS

MAX_SEARCHES = 3
search_count = 0

@tool
def web_search(query: str) -> str:
    """Search the web."""
    print(f"\nSEARCH CALLED: {query}")

    global search_count
    
    if search_count >= MAX_SEARCHES:
        return "Search limit reached. No further searches allowed."

    search_count += 1

    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                query,
                max_results=5
            )
        )

    return str(results)

# @tool
# def verify_claim(claim: str) -> str:
#     """
#     Verify a factual claim.
#     """

#     return web_search.invoke(claim)


def search_images(
    query: str
):

    with DDGS() as ddgs:

        results = list(
            ddgs.images(
                query,
                max_results=5
            )
        )

    return results

import pandas as pd


def generate_table_html(
    table_spec
):
    df = pd.DataFrame(
        table_spec["rows"],
        columns=table_spec["headers"]
    )

    return df.to_html(
        index=False
    )

def create_pdf(topic, analysis):

    filename = f"{topic}.pdf".replace(" ", "_")

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(topic, styles["Title"])
    )

    elements.append(Spacer(1, 12))

    for section, content in analysis.items():

        elements.append(
            Paragraph(
                section.replace("_", " ").title(),
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                str(content),
                styles["BodyText"]
            )
        )

        elements.append(Spacer(1, 8))

    doc.build(elements)

    return filename

