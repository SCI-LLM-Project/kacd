from pathlib import Path
from prompts_graphrag import graphrag_extraction_prompt, graphrag_community_prompt

import json

with open("variable_definitions/default_definitions.json", "r") as file:
    default = json.load(file)

with open("variable_definitions/ontological_definitions.json", "r") as file:
    ontology = json.load(file)


def prompt_er(entities, debug=False):
    system = (
        "\n--- ROLE ---\n"
        "You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged. "
        "You must output your final response in valid JSON, following this Pydantic Model:\n"
        """
class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged."
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )
        """
    )
    user = (
        "The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.\n"
        "Here are the rules for identifying duplicates:\n"
        "1. Entities with minor typographical differences should be considered duplicates.\n"
        "2. Entities with different formats but the same content should be considered duplicates.\n"
        "3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.\n"
        "4. If the entities refer to different numbers, dates, codes, medical ids, or products, do not merge entities.\n"
        "5. If the entities contain any numerical differences, do not merge entities.\n\n"
        "Here is the list of entities to process:\n"
        f"{entities}\n\n"
        "Please identify duplicates, merge them, and provide the merged list.\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages

def graph_extraction_prompt(text, debug=False):

    system_prompt = (
        "You are an expert in chronic lower back pain."
    )

    user = graphrag_extraction_prompt(text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages

def summarize_community(community, debug=False):
    system = (
        "You are an expert in chronic lower back pain."
    )

    user = (
        graphrag_community_prompt(community)
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages

def reduce(question, var1, var2, report, definitions, debug=False):
    system = (
        "\n--- ROLE ---\n"
        "You are an expert in the field of chronic pain, health care, and medicine. Given the question, output True or False. You will be reading two reports from the leading experts of chronic lower back pain to help answer the given question.\n"
        "Failure to follow directions will result in immediate termination.\n"
        "\n--- INSTRUCTIONS ---\n"
        "Use the relevant information in the report, as well as the definitions given, as context to help you answer the following True/False question.\n"
        "Begin by carefully breaking down the relevant information given in the report, noting any observations or conclusions that you can draw from the given information. "
        "Afterwards, identify any connections between the key points in the reports. Reread the report multiple times to make sure you identify all key details and all connections "
        "within the report. Then, think through the question step by step, thoroughly explaining your thought process. Use the information given in the report and your general "
        "knowledge to help you reason through the question and make your final answer. Remember, DO NOT base your conclusion solely on the report and DO NOT interpret the absence "
        "of information as evidence that the answer to the following question is false. If there is not enough information in the report, think through the question step by step "
        "again, using your general knowledge and reasoning skills to finalize your conclusion instead of relying solely on the report.\n"
        # "Think for a long time.\n"
        "\n--- OUTPUT STRUCTURE ---\n"
        "Based on your reasoning, output True or False as your final conclusion. Output your reasoning and conclusion explicitly in valid JSON. You must adhere to the specific format defined by this Pydantic Model:\n"
f"""
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: bool = Field(..., description="The culminating final conclusion or answer to the question")\n
"""
        "Specifically, each reasoning step must contain your intermediate thought process, and your final answer to the question must be in the conclusion field. "
        "Your final conclusion MUST be consistent with your reasoning.\n"

        "\n --- REPORT STRUCTURE ---\n"
        "You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of three types of relevant context:\n"
        "Knowledge Graph Communities: Each community will consist of a title, a summary, and a list of key findings.\n"
        "Chunks: Segments of text from the original source documents.\n"
        "Knowledge Graph Relationships: Facts extracted from the original source documents and organized into a relationship as apart of the Knowledge Graph. "
        "Each Relationship included will follow a knowledge graph triplet structure:\n"
        "('entity id': 'entity description')--['relationship type': 'relationship description']-->('entity id': 'entity description')\n"
        "Note that the reports will be given in markdown format."
    )
    user = (
        "\n--- VARIABLE DEFINITIONS ---\n"
        "Variable 1:\n"
        f"{ var1 }\n"
        "Definition of Variable 1:\n"
        f"{ definitions.get(var1, var1) }\n"
        "Variable 2:\n"
        f"{ var2 }\n"
        "Definition of Variable 2:\n"
        f"{ definitions.get(var2, var2) }\n"
        "\n--- REPORTS ---\n"
        f"{ report }\n\n"
        "\n--- QUESTION ---\n"
        f"{ question }\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages


def predict(question, var1, var2, definitions, debug=False):

    system = (
        "\n--- ROLE ---\n"
        "You are an expert in the field of chronic pain, health care, and medicine. Given the question, output True or False.\n"
        "Failure to follow directions will result in immediate termination.\n"
        "\n--- INSTRUCTIONS ---\n"
        "Think through the question step by step, thoroughly explaining your thought process. Use the definitions and your general knowledge to help you reason through the question and make your final answer.\n"
        # "Think for a long time.\n"
        "\n--- OUTPUT STRUCTURE ---\n"
        "Based on your reasoning, output True or False as your final conclusion. Output your reasoning and conclusion explicitly in valid JSON. You must adhere to the specific format defined by this Pydantic Model:\n"
f"""
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: bool = Field(..., description="The culminating final conclusion or answer to the question")\n
"""
        "Specifically, each reasoning step must contain your intermediate thought process, and your final answer to the question must be in the conclusion field. "
        "Your final conclusion MUST be consistent with your reasoning. "

    )

    user = (
        "\n--- VARIABLE DEFINITIONS ---\n"
        "Variable 1:\n"
        f"{ var1 }\n"
        "Definition of Variable 1:\n"
        f"{ definitions.get(var1, var1) }\n"
        "Variable 2:\n"
        f"{ var2 }\n"
        "Definition of Variable 2:\n"
        f"{ definitions.get(var2, var2) }\n"
        "\n--- QUESTION ---\n"
        f"{ question }\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages

def analyze_inconsistencies(query, reasoning, result, debug=False):
    system = (
        "\n--- TASK ---\n"
        "I am evaluating inconsistencies between the thought process of Large Language Models and their final answers to questions. "
        "You will be given a model's attempt to reason through the given question, and you will compare the reasoning process to its final conclusion. "
        "I want you to determine whether the final conclusion is consistent with the model's reasoning. Output True if the final conclusion is consistent with the model's reasoning "
        "or False if the model's final conclusion is inconsistent with the model's reasoning."
    )

    user = (
        "\n--- QUESTION --- \n"
        f"{query}\n"
        "\n--- REASONING --- \n"
        f"{reasoning.split('Conclusion:')[0]}\n"
        "\n--- FINAL CONCLUSION --- \n"
        f"{result}\n"
        "\n--- QUESTION ---\n"
        "Is the reasoning consistent with the final conclusion?"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages

def summarize(output, var1, var2, definitions, debug=False):
    system = (
        "\n--- ROLE ---\n"
        f"You are an expert in medicine and healthcare, aiming to summarize important documents to help physicians learn about { var1 }, and { var2 }, defined as {definitions.get(var1, var1)} and {definitions.get(var2, var2)}.\n"
        "You will be given output from a knowledge graph, consisting of entities, relationships, and summaries collected over the graph.\n"
        f"Focus on summarizing the information mainly within the relationships and the reports collected from the graph. Capture all details that may directly or indirectly relate {var1} to {var2}. \n"
        "Output only the summary and do not add any pre-amble, acknowledgements, or extraneous information such as citations. Failure to follow directions will result in immediate termination."
    )

    user = (
        "--- INSTRUCTIONS ---\n"
        "Given the output from the knowledge graph, summarize the output into a 750 word technical report.\n"
        f"Focus on extracting all details that pertain to { var1 }, { var2 }, and the relationships between { var1 } and { var2 },"
        "ignoring all other irrelevant pieces of information.\n"
        "\n--- KNOWLEDGE GRAPH OUTPUT ---\n"
        f"{ output }\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages

def reduce_rag(question, var1, var2, report, definitions, debug=False):
    system = (
        "\n--- ROLE ---\n"
        "You are an expert in the field of chronic pain, health care, and medicine. Given the question, output True or False. You will be reading two reports from the leading experts of chronic lower back pain to help answer the given question.\n"
        "Failure to follow directions will result in immediate termination.\n"
        "\n--- INSTRUCTIONS ---\n"
        "Use the relevant information in the report, as well as the definitions given, as context to help you answer the following True/False question.\n"
        "Begin by carefully breaking down the relevant information given in the report, noting any observations or conclusions that you can draw from the given information. "
        "Afterwards, identify any connections between the key points in the reports. Reread the report multiple times to make sure you identify all key details and all connections "
        "within the report. Then, think through the question step by step, thoroughly explaining your thought process. Use the information given in the report and your general "
        "knowledge to help you reason through the question and make your final answer. Remember, DO NOT base your conclusion solely on the report and DO NOT interpret the absence "
        "of information as evidence that the answer to the following question is false. If there is not enough information in the report, think through the question step by step "
        "again, using your general knowledge and reasoning skills to finalize your conclusion instead of relying solely on the report.\n"
        # "Think for a long time.\n"
        "\n--- OUTPUT STRUCTURE ---\n"
        "Based on your reasoning, output True or False as your final conclusion. Output your reasoning and conclusion explicitly in valid JSON. You must adhere to the specific format defined by this Pydantic Model:\n"
f"""
class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: bool = Field(..., description="The culminating final conclusion or answer to the question")\n
"""
        "Specifically, each reasoning step must contain your intermediate thought process, and your final answer to the question must be in the conclusion field. "
        "Your final conclusion MUST be consistent with your reasoning.\n"

        "\n --- REPORT STRUCTURE ---\n"
        "You will be given a report from experts of chronic lower back pain to help answer the given question. A report will consist of a collection of text segments relevant to the question given.\n"
        "Note that the reports will be given in markdown format."
    )
    user = (
        "\n--- VARIABLE DEFINITIONS ---\n"
        "Variable 1:\n"
        f"{ var1 }\n"
        "Definition of Variable 1:\n"
        f"{ definitions.get(var1, var1) }\n"
        "Variable 2:\n"
        f"{ var2 }\n"
        "Definition of Variable 2:\n"
        f"{ definitions.get(var2, var2) }\n"
        "\n--- REPORTS ---\n"
        f"{ report }\n\n"
        "\n--- QUESTION ---\n"
        f"{ question }\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if debug:
        print(messages)
    return messages
