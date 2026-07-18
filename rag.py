from sentence_transformers import SentenceTransformer
import faiss, glob, os
import numpy as np
import pandas as pd
import helpers
from prompts import *
import json


# %% [markdown]
# # Important Paremeters

# %%
path = "~/kg_aug_causal_disc_exp"

# %%
# variables of interest
with open("variable_definitions/default_definitions.json", "r") as file:
    def_map = json.load(file)

# %% [markdown]
# # Markdown parsing and chunking

# %%
import markdown_parser
from pathlib import Path
from config import DIRECTORY

chunks = []
files = Path(DIRECTORY).glob('**/*.md')
for file in files:
    print(file)
    if os.path.isfile(file):
        # simple markdown parser that removes citations, urls, references, acknowledgements, and basically everything after the conclusion
        content = markdown_parser.process_markdown_paper(str(file))
        # semantic chunk is chunking w.r.t sentences, and has overlap param as well
        chunks.extend(markdown_parser.semantic_chunk(content) )

# %% [markdown]
# # Creating the Retriever

# %%
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# %%
embs  = model.encode(chunks, convert_to_numpy=True)

# %%
norms = np.linalg.norm(embs, axis=1, keepdims=True)        # shape (N, 1)
embs_normalized = embs / np.clip(norms, a_min=1e-12, a_max=None)

# %%
dim   = embs_normalized.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs_normalized)

# %%
import numpy as np
from config import query_context_window
import helpers

# we aren't actually using k here
def get_k_docs(query: str, k: int = 100):
    # 1. Embed the query
    q_emb = model.encode([query], convert_to_numpy=True)  # shape (1, D)
    
    # 2. Search the FAISS index
    #    D: array of squared L2 distances, shape (1, k)
    #    I: array of indices of nearest neighbors, shape (1, k)
    D, I = index.search(q_emb, k)
    
    # 3. Fetch the top-k documents
    results = []
    token_count = 0
    for dist, idx in zip(D[0], I[0]):
        if helpers.token_count(chunks[idx] + "\n") + token_count > query_context_window:
            break
        token_count += helpers.token_count(chunks[idx] + "\n")
        results.append(chunks[idx]) # the original text or metadata
        

    return "\n".join(results)

# %% [markdown]
# # Setting Up RAG iterators

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
def local_retriever(query, var1, var2, summary, debug=False):
    if debug:
        print(reduce_rag(query, var1, var2, summary, def_map))
    response = generator(reduce_rag(query, var1, var2, summary, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})

    return response.conclusion, helpers.reasoning_to_string(response)
    

# %%
from promptsd.query_prompts import plausibility_prompt, temporality_prompt, causal_lit_prompt, association_prompt

def query_local_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2
        
    pquery = plausibility_prompt(var1, var2)
    preport = get_k_docs(pquery)
    aquery = association_prompt(var1, var2)
    areport = get_k_docs(aquery)
    tquery = temporality_prompt(var1, var2)
    treport = get_k_docs(tquery)
    plausibility, preasoning = local_retriever(pquery, var1, var2, preport)
    association, areasoning = local_retriever(aquery, var1, var2, areport)
    temporality, treasoning = local_retriever(tquery, var1, var2, treport)
    return [var1, var2, plausibility, preasoning, association, areasoning, temporality, treasoning, preport, areport, treport, label]

# %%
full = pd.read_csv(f"{path}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# %%
from tqdm import tqdm

tqdm.pandas()

# %% [markdown]
# # Setting Up the Experiment

# %%
res = full.progress_apply(query_local_causality, axis=1)

# %%
columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Plausibility Report", "Association Report", "Temporality Report", "Label"
local_res = pd.DataFrame(res.to_list(), columns=columns)
local_res.to_csv("results/rag.csv")
local_res

# %%
from sklearn.metrics import f1_score

f1_score(local_res["Label"], local_res["Plausibility"])

# %%
f1_score(local_res["Plausibility"], local_res["Label"])


