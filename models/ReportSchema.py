from pydantic import BaseModel, Field
from typing import Optional, List

class Finding(BaseModel):
    summary: str = Field(
        ...,
        description="Summary of the key insight about the community.",
        title="Finding Summary",
        min_length=1
    )
    explanation: str = Field(
        ...,
        description="A detailed explanation supporting the finding.",
        title="Finding Explanation",
        min_length=1
    )

class Report(BaseModel):
    title: str = Field(
        ...,
        description="community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.",
        title='Title',
        min_length=1
    )
    summary: str = Field(
        ...,
        description="An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.",
        title="Summary",
        min_length=1
    )
    impact_severity_rating: float = Field(
        ...,
        description="a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.",
        title="Impact Severity Rating"
    )
    rating_explanation: str = Field(
        ...,
        description="Give a single sentence explanation of the IMPACT severity rating.",
        title="Rating Explanation",
        min_length=1
    )
    detailed_findings: List[Finding] = Field(
        ...,
        description="A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.",
        title="Detailed Findings",
        min_length=5,
        max_length=10
    )
