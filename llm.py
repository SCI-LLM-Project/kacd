import outlines
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast, Callable
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

#from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.runnables import RunnableConfig
from prompts import *
from llm_client import get_client

def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {"description":node.description}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {"description":rel.description, "strength":rel.strength}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value

    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )

def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else None,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
            #description=el.description
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
            #description=el.description,
        )
        for el in rels
    ]

def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    parsed_schema,
) -> Tuple[List[Node], List[Relationship]]:
    nodes = (
        [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
        if parsed_schema.nodes
        else []
    )

    relationships = (
        [
            map_to_base_relationship(rel)
            for rel in parsed_schema.relationships
            if rel.type and rel.source_node_id and rel.target_node_id
        ]
        if parsed_schema.relationships
        else []
    )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class LLMGraphTransformer:

    def __init__(
        self,
        schema,
        prompt: Callable[[str, bool], str] = None,
        gleanings: int = 0
    ) -> None:
        
        self.prompt = prompt
        self.schema = schema
        self.gleanings = gleanings
        self.structured_llm = get_client(schema=self.schema)

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        res = self.structured_llm(self.prompt(text, debug=False), sampling_params={"n":1, "temperature":0, "top_k":1})
        # langchain backlips to convert the graph into langchain graph documents
        nodes, relationships = _convert_to_graph_document(res)
        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document, config) for document in documents]

    def convert_to_graph_documents_concurrent(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Same as convert_to_graph_documents, but dispatches every document's LLM
        call concurrently via the backend's .map() instead of looping one at a time.
        Worthwhile against a hosted API; a no-op-equivalent sequential fallback
        against the local vLLM server (see VLLMClient.map)."""
        all_messages = [self.prompt(document.page_content, debug=False) for document in documents]
        results = self.structured_llm.map(
            all_messages, sampling_params={"n": 1, "temperature": 0, "top_k": 1}
        )
        graph_documents = []
        for res, document in zip(results, documents):
            nodes, relationships = _convert_to_graph_document(res)
            graph_documents.append(
                GraphDocument(nodes=nodes, relationships=relationships, source=document)
            )
        return graph_documents

