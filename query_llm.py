# %%
import json
import warnings

import pandas as pd
from tqdm import tqdm

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
path = "~/kg_aug_causal_disc_exp"

# %% [markdown]
# ## Setting up LLM

# %%
from pydantic import BaseModel, Field
from typing import List

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: bool = Field(..., description="The culminating final conclusion or answer to the question")

# %%
from llm.factory import get_client

generator = get_client(schema=Answer)

# %%
# variables of interest
with open("variable_definitions/default_definitions.json", "r") as file:
    def_map = json.load(file)

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
full = pd.read_csv(f"{path}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Experiments

# %% [markdown]
# ## LLM

# %%
tqdm.pandas(desc="Querying LLM")
res = full.progress_apply(query_llm_causality, axis=1)

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Label"
llm_res = pd.DataFrame(res.to_list(), columns=columns)
llm_res.to_csv("results/llm.csv")
llm_res

# %%
from sklearn.metrics import f1_score
print("RESULTS FOR LLM ONLY")
print(f1_score(llm_res["Label"], llm_res["Plausibility"]))
