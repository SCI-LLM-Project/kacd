# %%
from neo4j import GraphDatabase, Result
from tqdm import tqdm
from typing import Dict, Any
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
import json

import pandas as pd

# %%
# user defined imports
from prompts.query_prompts.causal_literature_prompts import *
import util.helpers as helpers

# %%
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Parameters

# %%
path = "~/kg_aug_causal_disc_exp"

# %% [markdown]
# ## Setting Up Graph

# %%
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,NEO4J_DATABASE, DIRECTORY

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD),database=NEO4J_DATABASE)

def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

# %%
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    refresh_schema=False,
    driver_config={"notifications_disabled_classifications": ["DEPRECATION"]}
)

# %% [markdown]
# ## Setting up LLM

# %%
from pydantic import BaseModel, Field
from typing import List, Literal

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B', 'C']

# %%
from llm.factory import get_client

generator = get_client(schema=Answer)
summarizer = get_client()

# %% [markdown]
# # Local Retriever

# %%
# parameters for the local search query
from config import topChunks, topCommunities, topRels, topEntities

context = {}
lc_retrieval_query = helpers.load_query("local_search.cypher")

# %%
# variables of interest
with open("variable_definitions/default_definitions.json", "r") as file:
    def_map = json.load(file)

# %%
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

# %%
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

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

# %%
from util.helpers import *
from context_construction.build_context import stringify_report, format_triplet, construct_query_context
from config import query_context_window

# remember to clear the context cache when changing the summary prompt
context = {} # (var1, var2) -> summary

def retrieve_context_query(query) -> str:

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

    return construct_query_context(relationships, chunks, reports, max_context_window=8000)

# %%
def local_retriever(query, var1, var2, summary, debug=False):
    if debug:
        print(query_kg_causal_lit_prompt(query, var1, var2, summary, def_map))
        print(helpers.token_count(query_kg_causal_lit_prompt(query, var1, var2, summary, def_map)))
    response = generator(query_kg_causal_lit_prompt(query, var1, var2, summary, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})

    return response.conclusion, helpers.reasoning_to_string_multiple_choice(response)

# %%
def llm_retriever(query, var1, var2, debug=False):
    if debug:
        print(query_llm_causal_lit_prompt(query, var1, var2, def_map))
    response = generator(query_llm_causal_lit_prompt(query, var1, var2, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})
    return response.conclusion, helpers.reasoning_to_string_multiple_choice(response)

# %%
from prompts.query_prompts.metric_prompts import causal_lit_prompt

def query_local_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    clquery = causal_lit_prompt(var1, var2)
    clreport = retrieve_context_query(clquery)
    causal_literature, causal_lit_reasoning = local_retriever(clquery, var1, var2, clreport)
    return [var1, var2, causal_literature, causal_lit_reasoning, clreport, label]

# %%
def query_llm_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2
    
    clquery = causal_lit_prompt(var1, var2)
    causal_literature, causal_lit_reasoning = llm_retriever(clquery, var1, var2)
    return [var1, var2, causal_literature, causal_lit_reasoning, label]

# %%
full = pd.read_csv(f"{path}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %%
tqdm.pandas()

# %% [markdown]
# Experiments

# %% [markdown]
## LLM

# %%
res = full.progress_apply(query_llm_causality, axis=1)

# %%
columns = "Var1", "Var2", "Causal Literature",  "Causal Literature Reasoning", "Label"
llm_res = pd.DataFrame(res.to_list(), columns=columns)
llm_res.to_csv("results/llm_full_causal_literature.csv")
llm_res

# %% [markdown]
# ## RAG

# %% [markdown]
# ## Local Search

# %%
res = full.progress_apply(query_local_causality, axis=1)

# %%
columns = "Var1", "Var2", "Causal Literature",  "Causal Literature Reasoning", "Causal Literature Report", "Label"
local_res = pd.DataFrame(res.to_list(), columns=columns)
local_res.to_csv("results/kg+rag_full_causal_literature.csv")
local_res
