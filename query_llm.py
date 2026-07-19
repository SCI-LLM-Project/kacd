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
def query_llm_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2

    presponse = generator(query_llm_prompt(plausibility_prompt(var1, var2), var1, var2, def_map))
    aresponse = generator(query_llm_prompt(association_prompt(var1, var2), var1, var2, def_map))
    tresponse = generator(query_llm_prompt(temporality_prompt(var1, var2), var1, var2, def_map))

    return [var1, var2,
            presponse.conclusion, helpers.reasoning_to_string(presponse),
            aresponse.conclusion, helpers.reasoning_to_string(aresponse),
            tresponse.conclusion, helpers.reasoning_to_string(tresponse),
            label]

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
print("RESULTS FOR LLM ONLY")
print(f1_score(llm_res["Label"], llm_res["Plausibility"]))
