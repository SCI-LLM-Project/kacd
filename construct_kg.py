# %% [markdown]
# # This notebook is meant for extracting Knowledge Graphs from MMD Files

# %%
from llm.graph_transformer import LLMGraphTransformer
import util.helpers as helpers
import torch
import os
import pandas as pd
from langchain_community.graphs import Neo4jGraph

# %% [markdown]
# # Configure these parameters to select the appropriate graph and prompting strategy to use

# %%
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,NEO4J_DATABASE, DIRECTORY
graph = Neo4jGraph(NEO4J_URI,NEO4J_USERNAME,NEO4J_PASSWORD,NEO4J_DATABASE, refresh_schema=False)

### CLEARING ANY EXISTING GRAPH

# %%
# clear all nodes in the graph if not empty
graph.query("MATCH (n) DETACH DELETE n")

# %%
# Drop indexes
graph.query("DROP INDEX vector IF EXISTS")

# Drop all constraints
constraints = graph.query("SHOW CONSTRAINTS")
for c in constraints:
    graph.query(f"DROP CONSTRAINT {c['name']} IF EXISTS")

# Drop all indexes  
indexes = graph.query("SHOW INDEXES")
for idx in indexes:
    graph.query(f"DROP INDEX {idx['name']} IF EXISTS")

# %% [markdown]
# ## Markdown parsing and chunking

# %%
import util.markdown_parser as markdown_parser
from pathlib import Path

chunks = []
files = Path(DIRECTORY).glob('**/*.md')
for file in files:
    print(file)
    if os.path.isfile(file):
        # simple markdown parser that removes citations, urls, references, acknowledgements, and basically everything after the conclusion
        content = markdown_parser.process_markdown_paper(str(file))
        # semantic chunk is chunking w.r.t sentences, and has overlap param as well
        chunks.extend(markdown_parser.semantic_chunk(content) )

# %%
len(chunks)

# %% [markdown]
# ## Triplet Extraction

# %%
from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from retry import retry
from tqdm import tqdm
from models.KnowledgeGraphSchema import KnowledgeGraph
from prompts.construction_prompts.extraction_prompts import graph_extraction_prompt

llm_transformer = LLMGraphTransformer(
    schema=KnowledgeGraph,
    prompt=graph_extraction_prompt
)

# %%
# dispatches every chunk's extraction call concurrently via the backend's .map()
# instead of looping one chunk at a time - a big win against a hosted API, a no-op
# against the local vLLM server (see VLLMClient.map)
docs = [Document(page_content=chunk) for chunk in chunks]
graph_documents = llm_transformer.convert_to_graph_documents_concurrent(docs)

# %%
# this automatically combines nodes / relationships that have the same id
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

# %%
data = pd.DataFrame(graph.query("match (n:__Entity__)-[r]->(m:__Entity__) return n.id, n.description, r.strength, type(r), r.description, m.id, m.description order by type(r), n.id asc"))
data.to_csv("tmp/quick_analysis.csv")
data

# %% [markdown]
# # Entity Resolution

# %%
import util.helpers as helpers
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from graphdatascience import GraphDataScience

from prompts.construction_prompts.extraction_prompts import entity_resolution_prompt
from pydantic import BaseModel, create_model, Field
from typing import List, Optional
from retry import retry

# %%
pubmedbert_embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/S-PubMedBert-MS-MARCO",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# %%
graph.query("DROP INDEX vector IF EXISTS;")

# %%
vector = Neo4jVector.from_existing_graph(
    pubmedbert_embeddings,
    node_label='__Entity__',
    text_node_properties=['id', 'description'],
    embedding_node_property='embedding',
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# %%
# project graph
gds = GraphDataScience(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE

)

# Drop GDS projections if they exist
try:
    gds.graph.drop("entities")
except:
    pass
try:
    gds.graph.drop("communities")
except:
    pass

# %%
G, result = gds.graph.project(
    "entities",                   #  Graph name
    "__Entity__",                 #  Node projection
    "*",                          #  Relationship projection
    nodeProperties=["embedding"]  #  Configuration parameters
)

# %%
# experiment with a higher sim threshold
## Experimented with different thresholds, noted that most embeddings were highly similar were above 0.98. Here, we want to be extra conservative to prevent incorrect predictions
similarity_threshold = 0.98

gds.knn.mutate(
  G,
  nodeProperties=['embedding'],
  mutateRelationshipType= 'SIMILAR',
  mutateProperty= 'score',
  similarityCutoff=similarity_threshold,
  randomSeed=42,
  concurrency=1,
  sampleRate=1.0,
  deltaThreshold=0.0
)

# %%
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)

# %%
edit_distance_query = helpers.load_query("edit_distance.cypher")
print(edit_distance_query)

# %%
word_edit_distance = 2
potential_duplicate_candidates = graph.query(edit_distance_query, params={'distance': word_edit_distance, 'min_length': 5})

# %%
from prompts.construction_prompts.extraction_prompts import entity_resolution_prompt
from llm.factory import get_client

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )

extraction_llm = get_client(schema=Disambiguate)

@retry(tries=1, delay=2)
def entity_resolution(entities: List[str]) -> Optional[List[str]]:
    res = extraction_llm(entity_resolution_prompt(sorted(entities)), sampling_params={"n":1, "temperature":0, "top_k":1})
    return [
        el.entities
        for el in res.merge_entities
    ]

# %%
from tqdm import tqdm 

merged_entities = []

for el in tqdm(potential_duplicate_candidates, total=len(potential_duplicate_candidates), desc="Resolving entities"):
    merged_entities.extend(entity_resolution(el["combinedResult"]))

# %%
graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity__) WHERE e.id IN candidates
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
""", params={"data": merged_entities})

# %%
gds.graph.drop("entities")

# %%
G.drop()

# %% [markdown]
# # Communities

# %%
gds.graph.drop('communities')

# %%
G, result = gds.graph.project(
    "communities",  #  Graph name
    "__Entity__",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

# %%
wcc = gds.wcc.stats(G)
print(f"Component count: {wcc['componentCount']}")
print(f"Component distribution: {wcc['componentDistribution']}")

# %%
gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipWeightProperty="weight",
    randomSeed=42,
    concurrency=1,
    theta=0
)

# %%
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")

# %%
# creates community nodes and adds edges from each node to the community it belongs too. 
graph.query("""
MATCH (e:`__Entity__`)
UNWIND range(0, size(e.communities) - 1 , 1) AS index
CALL {
  WITH e, index
  WITH e, index
  WHERE index = 0
  MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET c.level = index
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
CALL {
  WITH e, index
  WITH e, index
  WHERE index > 0
  MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET current.level = index
  MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
  ON CREATE SET previous.level = index - 1
  MERGE (previous)-[:IN_COMMUNITY]->(current)
  RETURN count(*) AS count_1
}
RETURN count(*)
""")

# %%
graph.query("MATCH (n:`__Community__`) return count(n)")

# %%
# community structure
# finds the size of all communities with more than two entities
community_size_df = graph.query(
    """
    MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
    WITH c, count(distinct e) AS entities
    WHERE entities > 1
    RETURN split(c.id, '-')[0] AS level, entities
    """
)
community_size_df = pd.DataFrame(community_size_df)
helpers.community_analysis(community_size_df)

# %% [markdown]
# # Summarizing Communities

# %%
community_info = graph.query("""
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
WITH c, collect(e) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId, c.level as level,
       [r in relationships | {
                             start_id: startNode(r).id,
                             start_desc: startNode(r).description, 
                             rel_desc: r.description, 
                             rel_type: type(r),
                             end_id: endNode(r).id,
                             end_desc: endNode(r).description, 
                             degree: apoc.node.degree(startNode(r)) + apoc.node.degree(endNode(r))
                             }] AS triplets
""")

# %%
# first pass: sort the triplets, separate into two groups: leaves and nonleaves
context_window_limit = 8000

# %%
from build_context import split_and_sort, summarize_leaves, normalize_nonleaves, summarize_nonleaves

raw_leaves, raw_nonleaves = split_and_sort(community_info) 
leaves_with_reports, leaves_reports_map = summarize_leaves(raw_leaves, context_window_limit)

# %%
nonleaves_normalized = normalize_nonleaves(raw_nonleaves, leaves_reports_map)
nonleaves_with_reports = summarize_nonleaves(nonleaves_normalized, context_window_limit)

# %%
from build_context import normalize_summarized_community

leaves = list(
    map(normalize_summarized_community, leaves_with_reports)
)

nonleaves = list(
    map(normalize_summarized_community, nonleaves_with_reports)
)

# %%
# Store summaries
graph.query("""
UNWIND $data AS row
MERGE (c:__Community__ {id:row.community})
SET c.title=row.title, c.summary = row.summary, c.impact_severity_rating=row.impact_severity_rating, c.rating_explanation=row.rating_explanation, c.detailed_findings=row.detailed_findings
""", params={"data": leaves + nonleaves})

# %% [markdown]
# # Weighting the Graph

# %%
# community rank set to be the number of documents referenced by that community
graph.query("""
MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
""")


