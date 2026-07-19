"""
Runs the full pipeline (KG construction -> predictions -> consolidation ->
directed results) end to end as a plain script

    tmux new -s wflow
    python wflow.py
    # Ctrl-b d to detach; tmux attach -t wflow to reattach later

"""
import subprocess

import pandas as pd

from util.helpers import consolidate_causal_literature

# Generate KG
subprocess.run(["python", "construct_kg.py"], check=True)

# Generate Predictions
subprocess.run(["python", "query_kg.py"], check=True)
subprocess.run(["python", "query_llm.py"], check=True)
subprocess.run(["python", "query_rag.py"], check=True)
subprocess.run(["python", "query_causal_lit_kg.py"], check=True)
subprocess.run(["python", "query_causal_lit_llm.py"], check=True)
subprocess.run(["python", "query_causal_lit_rag.py"], check=True)

# Combine all results
kgrag = pd.read_csv("results/kgrag.csv", index_col=0).drop(columns=["Label"])
llm = pd.read_csv("results/llm.csv", index_col=0).drop(columns=["Label"])
rag = pd.read_csv("results/rag.csv", index_col=0).drop(columns=["Label"])

kgrag_cl = pd.read_csv("results/kg+rag_full_causal_literature.csv", index_col=0).drop(columns=["Label"])
llm_cl = pd.read_csv("results/llm_full_causal_literature.csv", index_col=0).drop(columns=["Label"])
llm_cl["Causal Literature Report"] = ''
rag_cl = pd.read_csv("results/llm+rag_full_causal_literature.csv", index_col=0).drop(columns=["Label"])

# Apply to all causal literature dataframes
kgrag_cl = consolidate_causal_literature(kgrag_cl)
llm_cl = consolidate_causal_literature(llm_cl)
rag_cl = consolidate_causal_literature(rag_cl)

kgrag = pd.merge(kgrag, kgrag_cl, on=["Var1", "Var2"])
rag = pd.merge(rag, rag_cl, on=["Var1", "Var2"])
llm = pd.merge(llm, llm_cl, on=["Var1", "Var2"])

# Add context column to each dataframe
kgrag['context'] = 'kg+llm_full.csv'
llm['context'] = 'llm_full.csv'
rag['context'] = 'rag_full.csv'

# Concatenate all dataframes
consolidated = pd.concat([kgrag, llm, rag], ignore_index=True)

consolidated.to_csv("results/results_undirected_combined.csv")

# Generate directed results
subprocess.run(["python", "query_directions_without_proto.py"], check=True)
subprocess.run(["python", "query_directions_with_proto.py"], check=True)
