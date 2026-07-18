#!/usr/bin/env python
# coding: utf-8

"""
Standalone script for summarizing communities from Neo4j graph.
Saves intermediate results to JSON for reloading if needed.
"""

import json
import os
from datetime import datetime
from langchain_community.graphs import Neo4jGraph
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

# Configuration
OUTPUT_DIR = "community_summaries"

def save_json(data, filename):
    """Save data to JSON file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"   Saved: {filepath}")
    return filepath

print("=" * 60)
print("Community Summarization Script")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Connect to Neo4j
print("\n1. Connecting to Neo4j database...")
graph = Neo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE, refresh_schema=False)
print("   Connected successfully!")

# Retrieve community information
print("\n2. Retrieving community information from graph...")
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
print(f"   Retrieved {len(community_info)} communities")

# Configuration
context_window_limit = 8000


# In[7]:


from build_context import split_and_sort, summarize_leaves, normalize_nonleaves, summarize_nonleaves

raw_leaves, raw_nonleaves = split_and_sort(community_info)
leaves_with_reports, leaves_reports_map = summarize_leaves(raw_leaves, context_window_limit)

# Save leaves_reports_map (convert to serializable format)
try:
    leaves_reports_serializable = {
        k: {
            "summary": v[0].model_dump() if hasattr(v[0], 'model_dump') else v[0].dict(),
            "summary_token_count": v[1],
            "community_token_count": v[2]
        }
        for k, v in leaves_reports_map.items()
    }
    save_json(leaves_reports_serializable, "leaves_reports_map.json")
except Exception as e:
    print(f"   Warning: Failed to save leaves_reports_map as JSON: {e}")


# In[8]:


nonleaves_normalized = normalize_nonleaves(raw_nonleaves, leaves_reports_map)
nonleaves_with_reports = summarize_nonleaves(nonleaves_normalized, context_window_limit)

# Save nonleaves_with_reports
try:
    nonleaves_serializable = [
        {
            "communityId": nl["communityId"],
            "level": nl["level"],
            "report": nl["Report"].model_dump() if hasattr(nl["Report"], 'model_dump') else nl["Report"].dict(),
            "community_token_count": nl.get("community_token_count")
        }
        for nl in nonleaves_with_reports
    ]
    save_json(nonleaves_serializable, "nonleaves_reports.json")
except Exception as e:
    print(f"   Warning: Failed to save nonleaves_with_reports as JSON: {e}")


# In[10]:


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

