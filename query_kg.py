# %%
import pandas as pd

# %%
# user defined imports
from prompts.query_prompts.base_prompts import *
import util.helpers as helpers

# %%
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## Parameters

# %%
from config import PROJECT_ROOT

# %% [markdown]
# ## Setting up LLM

# %%
from models.AnswerSchema import BooleanAnswer as Answer

# %%
from llm.factory import get_client

generator = get_client(schema=Answer)

# %% [markdown]
# # Local Retriever

# %%
# neo4j connection, vector index, embeddings, and the local-search context
# builder all live in the shared retriever module (set up once at import time)
from context_construction.retriever_kgrag import retrieve_kgrag_context

# %%
from config import def_map

# %%
def local_retriever(query, var1, var2, summary, debug=False):
    if debug:
        print(query_kg_prompt(query, var1, var2, summary, def_map))
    response = generator(query_kg_prompt(query, var1, var2, summary, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})

    return response.conclusion, helpers.reasoning_to_string(response)


# %%
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt

def query_local_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    pquery = plausibility_prompt(var1, var2)
    preport = retrieve_kgrag_context(pquery)

    aquery = association_prompt(var1, var2)
    areport = retrieve_kgrag_context(aquery)

    tquery = temporality_prompt(var1, var2)
    treport = retrieve_kgrag_context(tquery)

    plausibility, preasoning = local_retriever(pquery, var1, var2, preport)
    association, areasoning = local_retriever(aquery, var1, var2, areport)
    temporality, treasoning = local_retriever(tquery, var1, var2, treport)
    return [var1, var2, plausibility, preasoning, association, areasoning, temporality, treasoning, preport, areport, treport, label]

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Experiments

# %% [markdown]
# ## Local Search

# %%
res = helpers.parallel_apply(full, query_local_causality)

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Plausibility Report", "Association Report", "Temporality Report", "Label"
local_res = pd.DataFrame(res.to_list(), columns=columns)
local_res.to_csv("results/kgrag.csv")
local_res

# %%
from sklearn.metrics import f1_score
print("RESULTS FOR LOCAL")
print(f1_score(local_res["Label"], local_res["Plausibility"]))

# %%
print(local_res["Plausibility"].value_counts())
