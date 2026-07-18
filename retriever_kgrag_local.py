from build_context import stringify_report, format_triplet, construct_query_context
from config import *
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase, Result
from typing import Dict, Any
import torch
import pandas as pd
import helpers

# KG-RAG setup

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD),database=NEO4J_DATABASE)

def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    refresh_schema=False,
    driver_config={"notifications_disabled_classifications": ["DEPRECATION"]}
)

db_query(
    """
    CREATE VECTOR INDEX vector IF NOT EXISTS
    FOR (n:__Entity__)
    ON n.embedding
    OPTIONS {indexConfig: {
      `vector.dimensions`: 768,
      `vector.similarity_function`: "cosine"
    }};
    """
)

embedding = HuggingFaceEmbeddings(
    model_name="pritamdeka/S-PubMedBert-MS-MARCO",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# %%
lc_retrieval_query = helpers.load_query("local_search.cypher")
lc_vector = Neo4jVector.from_existing_index(
    embedding=embedding,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    index_name="vector", # may need to alter
    search_type="vector",
    # keyword_index_name="keyword",
    retrieval_query=lc_retrieval_query,
)

def retrieve_kgrag_context(query) -> str:

    res = lc_vector.similarity_search(
        query,
        k=topEntities,
        params={
            "topChunks": topChunks,
            "topCommunities": topCommunities,
            "topRels": topRels
        }
    )

    metadata = res[0].metadata
    reports = [stringify_report(report) for report in metadata["Reports"]]
    chunks = metadata["Chunks"]
    relationships = [format_triplet(triplet) for triplet in metadata["Relationships"]]

    return construct_query_context(relationships, chunks, reports, max_context_window=query_context_window)