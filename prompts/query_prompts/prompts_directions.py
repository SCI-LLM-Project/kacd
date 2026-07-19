from config import def_map

def DIRECT_CAUSAL_PROMPT(var1, var2):
    return f"""Question: Which causal direction is more likely?
A) {var1} causes {var2} - {var1} causes a change in {var2}, either directly or through one or more intermediate variables.
B) {var2} causes {var1} - {var2} causes a change in {var1}, either directly or through one or more intermediate variables.
    """

def create_plausibility_prompt_rag(var1, var2, report):
    """Create prompt for determining plausible causal direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which causal direction is more plausible, either directly or through one or more intermediate variables. Plausibility refers to the extent to which a hypothesized relationship may be biologically, theoretically, or scientifically reasonable, based on existing knowledge.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which causal relationship is more plausible?
A) {var1} causes {var2} - It is plausible that {var1} leads to changes in {var2}
B) {var2} causes {var1} - It is plausible that {var2} leads to changes in {var1}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of a collection of text segments relevant to the question given.
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]


def create_association_prompt_rag(var1, var2, report):
    """Create prompt for determining associational relationship direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which associational relationship direction is more likely. That is, if knowledge of the value of one variable gives information about the other.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which direction of association is more likely?
A) {var1} is associated with {var2} - Changes in {var1} are statistically associated with changes in {var2}
B) {var2} is associated with {var1} - Changes in {var2} are statistically associated with changes in {var1}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of a collection of text segments relevant to the question given.
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]


def create_temporality_prompt_rag(var1, var2, report):
    """Create prompt for determining temporal relationship direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which temporal causal relationship is more likely, either directly or through one or more intermediate variables.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which temporal causal relationship is more likely?
A) {var1} precedes {var2} - {var1} occurs or changes before {var2} in time
B) {var2} precedes {var1} - {var2} occurs or changes before {var1} in time

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of a collection of text segments relevant to the question given.
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]


def create_causal_prompt_rag(var1, var2, report):
    """Create prompt for determining causal direction (direct or indirect)."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which direction of the causal relationship is more likely.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

{DIRECT_CAUSAL_PROMPT(var1, var2)}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of a collection of text segments relevant to the question given.
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]

def create_plausibility_prompt_kg(var1, var2, report):
    """Create prompt for determining plausible causal direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which causal direction is more plausible, either directly or through one or more intermediate variables. Plausibility refers to the extent to which a hypothesized relationship may be biologically, theoretically, or scientifically reasonable, based on existing knowledge.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which causal relationship is more plausible?
A) {var1} causes {var2} - It is plausible that {var1} leads to changes in {var2}
B) {var2} causes {var1} - It is plausible that {var2} leads to changes in {var1}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of three types of relevant context:
Knowledge Graph Communities: Each community will consist of a title, a summary, and a list of key findings.
Chunks: Segments of text from the original source documents.
Knowledge Graph Relationships: Facts extracted from the original source documents and organized into a relationship as apart of the Knowledge Graph.
Each Relationship included will follow a knowledge graph triplet structure:
('entity id': 'entity description')--['relationship type': 'relationship description']-->('entity id': 'entity description')
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]


def create_association_prompt_kg(var1, var2, report):
    """Create prompt for determining associational relationship direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which associational relationship direction is more likely. That is, if knowledge of the value of one variable gives information about the other.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which direction of association is more likely?
A) {var1} is associated with {var2} - Changes in {var1} are statistically associated with changes in {var2}
B) {var2} is associated with {var1} - Changes in {var2} are statistically associated with changes in {var1}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of three types of relevant context:
Knowledge Graph Communities: Each community will consist of a title, a summary, and a list of key findings.
Chunks: Segments of text from the original source documents.
Knowledge Graph Relationships: Facts extracted from the original source documents and organized into a relationship as apart of the Knowledge Graph.
Each Relationship included will follow a knowledge graph triplet structure:
('entity id': 'entity description')--['relationship type': 'relationship description']-->('entity id': 'entity description')
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]


def create_temporality_prompt_kg(var1, var2, report):
    """Create prompt for determining temporal relationship direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which temporal causal relationship is more likely, either directly or through one or more intermediate variables.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which temporal causal relationship is more likely?
A) {var1} precedes {var2} - {var1} occurs or changes before {var2} in time
B) {var2} precedes {var1} - {var2} occurs or changes before {var1} in time

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of three types of relevant context:
Knowledge Graph Communities: Each community will consist of a title, a summary, and a list of key findings.
Chunks: Segments of text from the original source documents.
Knowledge Graph Relationships: Facts extracted from the original source documents and organized into a relationship as apart of the Knowledge Graph.
Each Relationship included will follow a knowledge graph triplet structure:
('entity id': 'entity description')--['relationship type': 'relationship description']-->('entity id': 'entity description')
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]


def create_causal_prompt_kg(var1, var2, report):
    """Create prompt for determining causal direction (direct or indirect)."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge, the following report, and the definitions, determine which direction of the causal relationship is more likely.

Report:
{report}

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

{DIRECT_CAUSAL_PROMPT(var1, var2)}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

--- REPORT STRUCTURE ---
You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of three types of relevant context:
Knowledge Graph Communities: Each community will consist of a title, a summary, and a list of key findings.
Chunks: Segments of text from the original source documents.
Knowledge Graph Relationships: Facts extracted from the original source documents and organized into a relationship as apart of the Knowledge Graph.
Each Relationship included will follow a knowledge graph triplet structure:
('entity id': 'entity description')--['relationship type': 'relationship description']-->('entity id': 'entity description')
Note that the reports will be given in markdown format.
    """

    return [{"role": "user", "content": content}]

# The LLM doesn't have a report, but I am including the report here so that I don't have to complicate the logic downstream
def create_plausibility_prompt_llm(var1, var2, report):
    """Create prompt for determining plausible causal direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge and the definitions, determine which causal direction is more plausible, either directly or through one or more intermediate variables. Plausibility refers to the extent to which a hypothesized relationship may be biologically, theoretically, or scientifically reasonable, based on existing knowledge.

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which causal relationship is more plausible?
A) {var1} causes {var2} - It is plausible that {var1} leads to changes in {var2}
B) {var2} causes {var1} - It is plausible that {var2} leads to changes in {var1}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")
    """

    return [{"role": "user", "content": content}]


def create_association_prompt_llm(var1, var2, report):
    """Create prompt for determining associational relationship direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge and the definitions, determine which associational relationship direction is more likely. That is, if knowledge of the value of one variable gives information about the other.

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which direction of association is more likely?
A) {var1} is associated with {var2} - Changes in {var1} are statistically associated with changes in {var2}
B) {var2} is associated with {var1} - Changes in {var2} are statistically associated with changes in {var1}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")
    """

    return [{"role": "user", "content": content}]


def create_temporality_prompt_llm(var1, var2, report):
    """Create prompt for determining temporal relationship direction."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge and the definitions, determine which temporal causal relationship is more likely, either directly or through one or more intermediate variables.

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

Question: Which temporal causal relationship is more likely?
A) {var1} precedes {var2} - {var1} occurs or changes before {var2} in time
B) {var2} precedes {var1} - {var2} occurs or changes before {var1} in time

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")
    """

    return [{"role": "user", "content": content}]


def create_causal_prompt_llm(var1, var2, report):
    """Create prompt for determining causal direction (direct or indirect)."""
    content = f"""You are an expert in the field of chronic pain, health care, and medicine. Based on your general knowledge and the definitions, determine which direction of the causal relationship is more likely.

Definition of Variable 1:
{ def_map.get(var1, var1) }

Definition of Variable 2:
{ def_map.get(var2, var2) }

{DIRECT_CAUSAL_PROMPT(var1, var2)}

Provide your reasoning and then give your final answer as either A or B. Your answer must be JSON adhering to this Pydantic Schema:
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")
    """

    return [{"role": "user", "content": content}]
