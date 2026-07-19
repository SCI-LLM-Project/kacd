# %%
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
from config import PROJECT_ROOT

# %% [markdown]
# ## Setting up LLM

# %%
from models.AnswerSchema import CausalLitAnswer as Answer

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
        print(query_kg_causal_lit_prompt(query, var1, var2, summary, def_map))
        print(helpers.token_count(query_kg_causal_lit_prompt(query, var1, var2, summary, def_map)))
    response = generator(query_kg_causal_lit_prompt(query, var1, var2, summary, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})

    return response.conclusion, helpers.reasoning_to_string_multiple_choice(response)

# %%
from prompts.query_prompts.metric_prompts import causal_lit_prompt

def query_local_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    clquery = causal_lit_prompt(var1, var2)
    clreport = retrieve_kgrag_context(clquery)
    causal_literature, causal_lit_reasoning = local_retriever(clquery, var1, var2, clreport)
    return [var1, var2, causal_literature, causal_lit_reasoning, clreport, label]

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Experiments

# %% [markdown]
# ## Local Search

# %%
res = helpers.parallel_apply(full, query_local_causality)

# %%
columns = "Var1", "Var2", "Causal Literature",  "Causal Literature Reasoning", "Causal Literature Report", "Label"
local_res = pd.DataFrame(res.to_list(), columns=columns)
local_res.to_csv("results/kg+rag_full_causal_literature.csv")
local_res
