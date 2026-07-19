# %%
import warnings

import pandas as pd
from sklearn.metrics import f1_score

# user defined imports
from prompts.query_prompts.base_prompts import *
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt
import util.helpers as helpers
from config import PROJECT_ROOT, def_map
from models.AnswerSchema import BooleanAnswer as Answer
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
# Phase 1: build all prompts up front (cheap, no I/O)
pairs = [
    # bandaid for now
    ("Sleep disturbance" if row["var1"] == "Sleep" else row["var1"],
     "Sleep disturbance" if row["var2"] == "Sleep" else row["var2"],
     row["label"])
    for _, row in full.iterrows()
]

pprompts = [query_llm_prompt(plausibility_prompt(var1, var2), var1, var2, def_map) for var1, var2, _ in pairs]
aprompts = [query_llm_prompt(association_prompt(var1, var2), var1, var2, def_map) for var1, var2, _ in pairs]
tprompts = [query_llm_prompt(temporality_prompt(var1, var2), var1, var2, def_map) for var1, var2, _ in pairs]

# %%
# Phase 2: one batched LLM call per metric via the client's .map() - concurrent,
# progress bar, and per-item failure isolation built in (a failed call comes back as None)
presponses = generator.map(pprompts)
aresponses = generator.map(aprompts)
tresponses = generator.map(tprompts)

# %%
# Phase 3: assemble rows. a failed call keeps its row, with the negative class
# (False) and empty reasoning - only the failed metric is affected, the row's
# other metrics keep their real answers
def conclusion_of(response):
    return response.conclusion if response is not None else False

def reasoning_of(response):
    return helpers.reasoning_to_string(response) if response is not None else ""

rows = [
    [var1, var2,
     conclusion_of(presponse), reasoning_of(presponse),
     conclusion_of(aresponse), reasoning_of(aresponse),
     conclusion_of(tresponse), reasoning_of(tresponse),
     label]
    for (var1, var2, label), presponse, aresponse, tresponse
    in zip(pairs, presponses, aresponses, tresponses)
]

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Label"
llm_res = pd.DataFrame(rows, columns=columns)
llm_res.to_csv("results/llm.csv")
llm_res

# %%
print("RESULTS FOR LLM ONLY")
print(f1_score(llm_res["Label"], llm_res["Plausibility"]))
