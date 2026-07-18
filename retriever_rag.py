import faiss
import util.helpers as helpers
import json
import util.markdown_parser as markdown_parser
import numpy as np
import os
import pandas as pd
from config import DIRECTORY, query_context_window
from pathlib import Path
from sentence_transformers import SentenceTransformer

chunks = []
files = Path(DIRECTORY).glob('**/*.md')
for file in files:
    print(file)
    if os.path.isfile(file):
        # simple markdown parser that removes citations, urls, references, acknowledgements, and basically everything after the conclusion
        content = markdown_parser.process_markdown_paper(str(file))
        # semantic chunk is chunking w.r.t sentences, and has overlap param as well
        chunks.extend(markdown_parser.semantic_chunk(content) )


model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# %%
embs  = model.encode(chunks, convert_to_numpy=True)

# %%
norms = np.linalg.norm(embs, axis=1, keepdims=True)        # shape (N, 1)
embs_normalized = embs / np.clip(norms, a_min=1e-12, a_max=None)

dim   = embs_normalized.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs_normalized)

def retrieve_rag_context(query: str, k: int = 100):
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