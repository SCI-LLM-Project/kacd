# %%
from sentence_transformers import SentenceTransformer
import faiss, glob, os
import numpy as np
import pandas as pd
import util.helpers as helpers

# %% [markdown]
# # Important Paremeters

# %%
from config import PROJECT_ROOT

# %%
from config import def_map

# %% [markdown]
# # Markdown parsing and chunking

# %%
import util.markdown_parser as markdown_parser
from pathlib import Path
from config import DIRECTORY

chunks = []
files = Path(DIRECTORY).glob('**/*.md')
for file in files:
    print(file)
    if os.path.isfile(file):
        # just reads markdown file into a string (reference stripping is disabled)
        content = markdown_parser.process_markdown_paper(str(file))
        # semantic chunk is chunking w.r.t sentences, and has overlap param as well
        chunks.extend(markdown_parser.semantic_chunk(content))

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
import util.helpers as helpers

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
from models.AnswerSchema import CausalLitAnswer as Answer

# %%
from llm.factory import get_client
generator = get_client(schema=Answer)

# %%
from prompts.query_prompts.causal_literature_prompts import query_rag_causal_lit_prompt

def rag_retriever(query, var1, var2, summary, debug=False):
    if debug:
        print(query_rag_causal_lit_prompt(query, var1, var2, summary, def_map))
    response = generator(query_rag_causal_lit_prompt(query, var1, var2, summary, def_map), sampling_params={"n":1, "temperature":0.0, "top_k":1})

    return response.conclusion, helpers.reasoning_to_string_multiple_choice(response)
    

# %%
from prompts.query_prompts.metric_prompts import causal_lit_prompt

def query_rag_causality(row):
    var1, var2, label = row['var1'], row['var2'], row["label"]
    # bandaid for now
    var1 = "Sleep disturbance" if var1 == "Sleep" else var1
    var2 = "Sleep disturbance" if var2 == "Sleep" else var2
        
    clquery = causal_lit_prompt(var1, var2)
    clreport = get_k_docs(clquery)
    causal_lit, clreasoning = rag_retriever(clquery, var1, var2, clreport)
    return [var1, var2, causal_lit, clreasoning, clreport, label]

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


