"""
Runs the full pipeline (KG construction -> predictions -> consolidation ->
directed results) end to end as a plain script

    tmux new -s wflow
    python wflow.py
    # Ctrl-b d to detach; tmux attach -t wflow to reattach later

"""
import os
import subprocess
from datetime import datetime

import pandas as pd

from util.helpers import consolidate_causal_literature

LOG_PATH = "log/log.out"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def run(script):
    """Run a pipeline script with its stdout and stderr appended to LOG_PATH
    under a timestamped header. Nothing shows in the terminal while a script
    runs - follow along with `tail -f log/log.out`."""
    with open(LOG_PATH, "a") as log:
        log.write(f"\n===== {script} @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        log.flush()
        # -u: unbuffered, so prints land in the log as they happen rather than
        # in one block when the script exits
        subprocess.run(["python", "-u", script], check=True, stdout=log, stderr=subprocess.STDOUT)

# Generate KG
run("construct_kg.py")

# Generate Predictions
run("query_kg.py")
run("query_llm.py")
run("query_rag.py")
run("query_causal_lit_kg.py")
run("query_causal_lit_llm.py")
run("query_causal_lit_rag.py")

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
run("query_directions_without_proto.py")
run("query_directions_with_proto.py")
