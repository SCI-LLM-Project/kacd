from pydantic import BaseModel, Field
from typing import List, Literal

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class BooleanAnswer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: bool = Field(..., description="The culminating final conclusion or answer to the question")

class CausalLitAnswer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B', 'C'] = Field(..., description="The culminating final conclusion or answer to the question")

class DirectionAnswer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")
