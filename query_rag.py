# %%
import pandas as pd
import util.helpers as helpers
from prompts.query_prompts.base_prompts import *


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
from models.AnswerSchema import BooleanAnswer as Answer

# %%
from llm.factory import get_client
generator = get_client(schema=Answer)

# %%
def rag_retriever(query, var1, var2, summary, debug=False):
    if debug:
        print(query_rag_prompt(query, var1, var2, summary, def_map))
    response = generator(query_rag_prompt(query, var1, var2, summary, def_map))

    return response.conclusion, helpers.reasoning_to_string(response)


# %%
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt

def query_rag_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    pquery = plausibility_prompt(var1, var2)
    preport = retrieve_rag_context(pquery)
    aquery = association_prompt(var1, var2)
    areport = retrieve_rag_context(aquery)
    tquery = temporality_prompt(var1, var2)
    treport = retrieve_rag_context(tquery)
    plausibility, preasoning = rag_retriever(pquery, var1, var2, preport)
    association, areasoning = rag_retriever(aquery, var1, var2, areport)
    temporality, treasoning = rag_retriever(tquery, var1, var2, treport)
    return [var1, var2, plausibility, preasoning, association, areasoning, temporality, treasoning, preport, areport, treport, label]

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Setting Up the Experiment

# %%
res = helpers.parallel_apply(full, query_rag_causality)

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Plausibility Report", "Association Report", "Temporality Report", "Label"
rag_res = pd.DataFrame(res.to_list(), columns=columns)
rag_res.to_csv("results/rag.csv")
rag_res

# %%
from sklearn.metrics import f1_score

f1_score(rag_res["Label"], rag_res["Plausibility"])

# %%
f1_score(rag_res["Plausibility"], rag_res["Label"])
