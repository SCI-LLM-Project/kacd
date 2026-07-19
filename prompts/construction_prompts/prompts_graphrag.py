# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""
from config import def_map

def graphrag_extraction_prompt(input):
    GRAPH_EXTRACTION_PROMPT = f"""
    -Goal-
    Given a text document that is potentially relevant to this activity, identify all entities from the text and all relationships among the identified entities.
    Format your output following this pydantic schema:
    class KnowledgeGraph(BaseModel):
        nodes: Optional[List[SimpleNode]] = Field(
            ...,
            description='List of nodes',
            title='Nodes'
        )
        relationships: Optional[List[SimpleRelationship]] = Field(
            ...,
            description='List of relationships',
            title='Relationships'
        )

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity_name: Name of the entity, capitalized
    - entity_type: Type of the entity, capitalized
    - entity_description: Comprehensive description of the entity's attributes and activities

    Specifically, focus on extracting entities pertaining to the following terms:
    - {def_map.get('Sex')}
    - {def_map.get('PEG')}
    - {def_map.get('Sleep disturbance')}
    - {def_map.get('Depression')}
    - {def_map.get('Anxiety')}
    - {def_map.get('Obesity')}
    - {def_map.get('Alcohol')}
    - {def_map.get('Fear_avoidance')}
    - {def_map.get('Catastrophizing')}
    - {def_map.get('Education')}
    - {def_map.get('Financial_level')}
    - {def_map.get('Age')}
    - {def_map.get('Smoking')}
    - {def_map.get('CCI')}

    Format each entity following this pydantic schema:
    class SimpleNode(BaseModel):
        id: str = Field(
            ...,
            description='Name of the entity, capitalized',
            title='Id',
            min_length=1
        )
        type: str = Field(
            ...,
            description='Type of the entity, capitalized',
            title='Type',
            min_length=1
        )
        description: str = Field(
            ...,
            description="Comprehensive description of the entity's attributes and activities",
            title='Description',
            min_length=1
        )
    
    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
    For each pair of related entities, extract the following information:
    - source_entity: name of the source entity, as identified in step 1
    - target_entity: name of the target entity, as identified in step 1
    - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
    - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
    - relationship_type: name of the relationship
    
    Format each relationship following this pydantic schema:
    class SimpleRelationship(BaseModel):
        source_node_id: str = Field(
            ...,
            description='name of the source entity',
            title='Source Node Id',
            min_length=1
        )
        source_node_type: str = Field(
            ...,
            description="type of the source entity",
            title='Source Node Type',
            min_length=1
        )
        target_node_id: str = Field(
            ...,
            description='name of the target entity',
            title='Target Node Id',
            min_length=1
        )
        target_node_type: str = Field(
            ...,
            description="type of the target entity",
            title='Target Node Type',
            min_length=1
        )
        type: str = Field(
            ...,
            description="name of the relationship",
            title='Type',
            min_length=1
        )
        strength: int = Field(
            ...,
            description="a numeric score indicating strength of the relationship between the source entity and target entity",
            title='Strength'
        )
        description: str = Field(
            ...,
            description='explanation as to why you think the source entity and the target entity are related to each other',
            title='Description',
            min_length=1
        )
    
    -Real Data-
    ######################
    Text: {input}
    ######################
    Output:"""

    return GRAPH_EXTRACTION_PROMPT

def graphrag_community_prompt(text, max_report_length=2000):
    COMMUNITY_PROMPT = f"""
    You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

    # Goal
    Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

    # Given Input Structure
    Each entity and relationship included will follow a knowledge graph triplet structure:
    ('entity id': 'entity description')--['relationship type': 'relationship description']-->('entity id': 'entity description')

    In addition, you may be given reports from other subcommunities to analyze as well, following the same pydantic structure outlined below.

    # Report Structure

    The report should include the following sections:

    - TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
    - SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
    - IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
    - RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
    - DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

    Format the report following this pydantic schema:
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

    Do not include information where the supporting evidence for it is not provided.
    Limit the total report length to {max_report_length} words.

    # Real Data

    Use the following text for your answer. Do not make anything up in your answer.

    Text:
    {text}
    """
    return COMMUNITY_PROMPT

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
