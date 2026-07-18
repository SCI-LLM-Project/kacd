from construction_prompts.prompts_graphrag import graphrag_extraction_prompt, graphrag_community_prompt


def entity_resolution_prompt(entities, debug=False):
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
