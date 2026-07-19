# %%
import warnings

import pandas as pd

# %%
# user defined imports
from prompts.query_prompts.causal_literature_prompts import *
from prompts.query_prompts.metric_prompts import causal_lit_prompt
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
from models.AnswerSchema import CausalLitAnswer as Answer

# %%
from llm.factory import get_client

generator = get_client(schema=Answer)

# %%
from config import def_map

# %%
def llm_retriever(query, var1, var2, debug=False):
    if debug:
        print(query_llm_causal_lit_prompt(query, var1, var2, def_map))
    response = generator(query_llm_causal_lit_prompt(query, var1, var2, def_map))
    return response.conclusion, helpers.reasoning_to_string_multiple_choice(response)

# %%
def query_llm_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    clquery = causal_lit_prompt(var1, var2)
    causal_literature, causal_lit_reasoning = llm_retriever(clquery, var1, var2)
    return [var1, var2, causal_literature, causal_lit_reasoning, label]

# %%
full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %% [markdown]
# # Experiments

# %% [markdown]
# ## LLM

# %%
res = helpers.parallel_apply(full, query_llm_causality)

# %%
columns = "Var1", "Var2", "Causal Literature",  "Causal Literature Reasoning", "Label"
llm_res = pd.DataFrame(res.to_list(), columns=columns)
llm_res.to_csv("results/llm_full_causal_literature.csv")
llm_res
