# %%
import warnings

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

# user defined imports
from prompts.query_prompts.base_prompts import *
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt
import util.helpers as helpers
from config import PROJECT_ROOT, def_map
from models.AnswerSchema import BooleanAnswer as Answer
from llm.factory import get_client
# neo4j connection, vector index, embeddings, and the local-search context
# builder all live in the shared retriever module (set up once at import time)
from context_construction.retriever_kgrag import retrieve_kgrag_context

# %%
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Setting up LLM

# %%
generator = get_client(schema=Answer)

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Experiments

# %% [markdown]
# ## Local Search

# %%
# Phase 1: build all queries and retrieve all contexts (no LLM calls).
pairs = [
    # bandaid for now
    ("Sleep disturbance" if row["var1"] == "Sleep" else row["var1"],
     "Sleep disturbance" if row["var2"] == "Sleep" else row["var2"],
     row["label"])
    for _, row in full.iterrows()
]

pqueries = [plausibility_prompt(var1, var2) for var1, var2, _ in pairs]
aqueries = [association_prompt(var1, var2) for var1, var2, _ in pairs]
tqueries = [temporality_prompt(var1, var2) for var1, var2, _ in pairs]

# %%
preports = [retrieve_kgrag_context(query) for query in tqdm(pqueries, desc="Plausibility retrieval")]
areports = [retrieve_kgrag_context(query) for query in tqdm(aqueries, desc="Association retrieval")]
treports = [retrieve_kgrag_context(query) for query in tqdm(tqueries, desc="Temporality retrieval")]

# %%
# Phase 2: one batched LLM call per metric via the client's .map() - concurrent,
# progress bar, and per-item failure isolation built in (a failed call comes back as None)
pprompts = [
    query_kg_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, pqueries, preports)
]
aprompts = [
    query_kg_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, aqueries, areports)
]
tprompts = [
    query_kg_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, tqueries, treports)
]

presponses = generator.map(pprompts)
aresponses = generator.map(aprompts)
tresponses = generator.map(tprompts)

# %%
# Phase 3: assemble rows. a failed call keeps its row, with the negative class
# (False) and empty reasoning - only the failed metric is affected, the row's
# other metrics keep their real answers
rows = [
    [var1, var2,
     helpers.conclusion_of(presponse), helpers.reasoning_of(presponse),
     helpers.conclusion_of(aresponse), helpers.reasoning_of(aresponse),
     helpers.conclusion_of(tresponse), helpers.reasoning_of(tresponse),
     preport, areport, treport, label]
    for (var1, var2, label), presponse, aresponse, tresponse, preport, areport, treport
    in zip(pairs, presponses, aresponses, tresponses, preports, areports, treports)
]

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Plausibility Report", "Association Report", "Temporality Report", "Label"
local_res = pd.DataFrame(rows, columns=columns)
local_res.to_csv("results/kgrag.csv")
local_res

# %%
print("RESULTS FOR LOCAL")
print(f1_score(local_res["Label"], local_res["Plausibility"]))

# %%
print(local_res["Plausibility"].value_counts())
