# %%
import warnings

import pandas as pd

# user defined imports
from prompts.query_prompts.causal_literature_prompts import *
from prompts.query_prompts.metric_prompts import causal_lit_prompt
import util.helpers as helpers
from config import PROJECT_ROOT, def_map
from models.AnswerSchema import CausalLitAnswer as Answer
from llm.factory import get_client

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
# ## LLM

# %%
# build every pair's prompt up front (cheap, no I/O), then dispatch all the LLM
# calls in one batch via the client's .map() - concurrent, progress bar, and
# per-item failure isolation built in (a failed call comes back as None)
pairs = [
    # bandaid for now
    ("Sleep disturbance" if row["var1"] == "Sleep" else row["var1"],
     "Sleep disturbance" if row["var2"] == "Sleep" else row["var2"],
     row["label"])
    for _, row in full.iterrows()
]

prompts = [
    query_llm_causal_lit_prompt(causal_lit_prompt(var1, var2), var1, var2, def_map)
    for var1, var2, _ in pairs
]

# %%
responses = generator.map(prompts)

# %%
# a failed call keeps its row: 'C' (no causal relationship) with empty reasoning
rows = [
    [var1, var2,
     helpers.conclusion_of(response, default="C"),
     helpers.reasoning_of(response, to_string=helpers.reasoning_to_string_multiple_choice),
     label]
    for (var1, var2, label), response in zip(pairs, responses)
]

# %%
columns = "Var1", "Var2", "Causal Literature",  "Causal Literature Reasoning", "Label"
llm_res = pd.DataFrame(rows, columns=columns)
llm_res.to_csv("results/llm_full_causal_literature.csv")
llm_res
