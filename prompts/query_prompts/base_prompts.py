def query_kg_prompt(question, var1, var2, report, definitions, debug=False):
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
        "Knowledge Graph Communities: Each community will consist of a title, a summary, a list of key findings, and an impact rating with an explanation.\n"
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


def query_llm_prompt(question, var1, var2, definitions, debug=False):

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

def query_rag_prompt(question, var1, var2, report, definitions, debug=False):
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
