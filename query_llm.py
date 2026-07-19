# %%
import warnings

import pandas as pd

# %%
# user defined imports
from prompts.query_prompts.base_prompts import *
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt
import util.helpers as helpers

# %%
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

# %%
from config import def_map

# %%
def llm_retriever(query, var1, var2, debug=False):
    if debug:
        print(query_llm_prompt(query, var1, var2, def_map))
    response = generator(query_llm_prompt(query, var1, var2, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})
    return response.conclusion, helpers.reasoning_to_string(response)

# %%
def query_llm_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    pquery = plausibility_prompt(var1, var2)
    aquery = association_prompt(var1, var2)
    tquery = temporality_prompt(var1, var2)
    plausibility, preasoning = llm_retriever(pquery, var1, var2)
    association, areasoning = llm_retriever(aquery, var1, var2)
    temporality, treasoning = llm_retriever(tquery, var1, var2)
    return [var1, var2, plausibility, preasoning, association, areasoning, temporality, treasoning, label]

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Experiments

# %% [markdown]
# ## LLM

# %%
res = helpers.parallel_apply(full, query_llm_causality)

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Label"
llm_res = pd.DataFrame(res.to_list(), columns=columns)
llm_res.to_csv("results/llm.csv")
llm_res

# %%
from sklearn.metrics import f1_score
print("RESULTS FOR LLM ONLY")
print(f1_score(llm_res["Label"], llm_res["Plausibility"]))
