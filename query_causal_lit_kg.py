# %%
import warnings

import pandas as pd
from tqdm import tqdm

# user defined imports
from prompts.query_prompts.causal_literature_prompts import *
from prompts.query_prompts.metric_prompts import causal_lit_prompt
import util.helpers as helpers
from config import PROJECT_ROOT, def_map, RESULT_DIR
from models.AnswerSchema import CausalLitAnswer as Answer
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

clqueries = [causal_lit_prompt(var1, var2) for var1, var2, _ in pairs]

# %%
clreports = [retrieve_kgrag_context(query) for query in tqdm(clqueries, desc="Causal literature retrieval")]

# %%
# Phase 2: one batched LLM call via the client's .map() - concurrent, progress
# bar, and per-item failure isolation built in (a failed call comes back as None)
clprompts = [
    query_kg_causal_lit_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, clqueries, clreports)
]

clresponses = generator.map(clprompts)

# %%
# Phase 3: assemble rows. a failed call keeps its row, with the negative class
# ('C', no causal relationship) and empty reasoning
rows = [
    [var1, var2,
     helpers.conclusion_of(clresponse, default="C"),
     helpers.reasoning_of(clresponse, to_string=helpers.reasoning_to_string_multiple_choice),
     clreport, label]
    for (var1, var2, label), clresponse, clreport in zip(pairs, clresponses, clreports)
]

# %%
columns = "Var1", "Var2", "Causal Literature",  "Causal Literature Reasoning", "Causal Literature Report", "Label"
local_res = pd.DataFrame(rows, columns=columns)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
local_res.to_csv(RESULT_DIR / "kg+rag_full_causal_literature.csv")
local_res
