# %%
import pandas as pd
from sklearn.metrics import f1_score

import util.helpers as helpers
from prompts.query_prompts.base_prompts import *
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt
from config import PROJECT_ROOT, def_map
# corpus chunking, embedding, and the FAISS index all live in the shared
# retriever module (built once at import time)
from context_construction.retriever_rag import retrieve_rag_context
from models.AnswerSchema import BooleanAnswer as Answer
from llm.factory import get_client

# %% [markdown]
# # Setting Up RAG iterators

# %%
generator = get_client(schema=Answer)

# %%
def query_rag_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    pquery = plausibility_prompt(var1, var2)
    preport = retrieve_rag_context(pquery)
    presponse = generator(query_rag_prompt(pquery, var1, var2, preport, def_map))

    aquery = association_prompt(var1, var2)
    areport = retrieve_rag_context(aquery)
    aresponse = generator(query_rag_prompt(aquery, var1, var2, areport, def_map))

    tquery = temporality_prompt(var1, var2)
    treport = retrieve_rag_context(tquery)
    tresponse = generator(query_rag_prompt(tquery, var1, var2, treport, def_map))

    return [var1, var2,
            presponse.conclusion, helpers.reasoning_to_string(presponse),
            aresponse.conclusion, helpers.reasoning_to_string(aresponse),
            tresponse.conclusion, helpers.reasoning_to_string(tresponse),
            preport, areport, treport, label]

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
f1_score(rag_res["Label"], rag_res["Plausibility"])

# %%
f1_score(rag_res["Plausibility"], rag_res["Label"])
