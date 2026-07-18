from pydantic import BaseModel, Field
from typing import Optional, List

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
