# %%
import pandas as pd
import util.helpers as helpers

# %% [markdown]
# # Important Paremeters

# %%
from config import PROJECT_ROOT

# %%
from config import def_map

# %% [markdown]
# # Retriever

# %%
# corpus chunking, embedding, and the FAISS index all live in the shared
# retriever module (built once at import time)
from context_construction.retriever_rag import retrieve_rag_context

# %% [markdown]
# # Setting Up RAG iterators

# %%
from models.AnswerSchema import CausalLitAnswer as Answer

# %%
from llm.factory import get_client
generator = get_client(schema=Answer)

# %%
from prompts.query_prompts.causal_literature_prompts import query_rag_causal_lit_prompt
from prompts.query_prompts.metric_prompts import causal_lit_prompt

def query_rag_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    clquery = causal_lit_prompt(var1, var2)
    clreport = retrieve_rag_context(clquery)
    clresponse = generator(query_rag_causal_lit_prompt(clquery, var1, var2, clreport, def_map))
    return [var1, var2, clresponse.conclusion, helpers.reasoning_to_string_multiple_choice(clresponse), clreport, label]

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Setting Up the Experiment

# %%
res = helpers.parallel_apply(full, query_rag_causality)

# %%
columns = "Var1", "Var2", "Causal Literature", "Causal Literature Reasoning", "Causal Literature Report", "Label"
rag_res = pd.DataFrame(res.to_list(), columns=columns)
rag_res.to_csv("results/llm+rag_full_causal_literature.csv")
rag_res
