# %% [markdown]
# # KG+RAG pipeline: fresh kgrag.csv -> directed results
#
# Runs the KG+RAG arm of the pipeline end to end: regenerates the undirected
# KG-RAG predictions (kgrag.csv), consolidates them with the causal-literature
# results, and resolves directions - both without and with the proto model. The
# three contexts wflow.py processes (kg+llm, llm, rag) are narrowed to just
# KG-RAG here; the existing query/direction scripts are left untouched and
# their outputs in results/ are not overwritten.
#
# Steps:
#   1. regenerate the undirected KG-RAG predictions (same retrieve -> prompt ->
#      score flow as query_kg.py), written to the output dir as kgrag.csv
#   2. consolidate the fresh kgrag.csv with the EXISTING
#      results/<RUN_NAME>/kg+rag_full_causal_literature.csv (from
#      query_causal_lit_kg.py - not regenerated here) into an undirected
#      combined frame, mirroring the "Combine all results" section of wflow.py
#   3. resolve bidirectional edges WITHOUT proto (Plausibility / Association /
#      Temporality; causal literature stands alone and already accounts for
#      direction, so it is skipped - same as query_directions_without_proto.py)
#   4. resolve bidirectional edges WITH proto (the three metrics plus causal
#      literature, ORed with the proto edges from data/proto/proto.csv - same
#      as query_directions_with_proto.py)
#
# Everything lands in tmp/kgrag_pipeline_<timestamp>/ - tmp/ is covered by
# .gitignore, so nothing here can end up committed.
#
# Prerequisites: neo4j running in docker, LLM_API_KEY in .env,
# results/<RUN_NAME>/kg+rag_full_causal_literature.csv present, run from the
# project root:
#   python run_kgrag_pipeline.py

# %%
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import config
from config import RESULT_DIR, PROTO_DIR, PROJECT_ROOT, def_map
import util.helpers as helpers
from util.helpers import consolidate_causal_literature
from llm.factory import get_client
from prompts.query_prompts.base_prompts import query_kg_prompt
from prompts.query_prompts.metric_prompts import plausibility_prompt, temporality_prompt, association_prompt
from prompts.query_prompts.prompts_directions import DIRECT_CAUSAL_PROMPT, create_causal_prompt_kg
from context_construction.retriever_kgrag import retrieve_kgrag_context
from models.AnswerSchema import BooleanAnswer, DirectionAnswer

# %%
OUT_DIR = PROJECT_ROOT / "tmp" / f"kgrag_pipeline_{time.strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"writing outputs to {OUT_DIR}")

# %% [markdown]
# ## Step 1: regenerate undirected KG-RAG predictions (kgrag.csv)
#
# Same three-phase flow as query_kg.py: retrieve all contexts, one batched LLM
# call per metric, assemble rows - only the output location differs.

# %%
answer_generator = get_client(schema=BooleanAnswer)

full = pd.read_csv(f"{PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# Phase 1: build all queries and retrieve all contexts (no LLM calls).
pairs = [
    # bandaid for now
    ("Sleep disturbance" if row["var1"] == "Sleep" else row["var1"],
     "Sleep disturbance" if row["var2"] == "Sleep" else row["var2"],
     row["label"])
    for _, row in full.iterrows()
]

pqueries = [plausibility_prompt(var1, var2) for var1, var2, _ in pairs]
aqueries = [association_prompt(var1, var2) for var1, var2, _ in pairs]
tqueries = [temporality_prompt(var1, var2) for var1, var2, _ in pairs]

preports = [retrieve_kgrag_context(query) for query in tqdm(pqueries, desc="Plausibility retrieval")]
areports = [retrieve_kgrag_context(query) for query in tqdm(aqueries, desc="Association retrieval")]
treports = [retrieve_kgrag_context(query) for query in tqdm(tqueries, desc="Temporality retrieval")]

# Phase 2: one batched LLM call per metric via the client's .map() - concurrent,
# progress bar, and per-item failure isolation built in (a failed call comes back as None)
pprompts = [
    query_kg_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, pqueries, preports)
]
aprompts = [
    query_kg_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, aqueries, areports)
]
tprompts = [
    query_kg_prompt(query, var1, var2, report, def_map)
    for (var1, var2, _), query, report in zip(pairs, tqueries, treports)
]

presponses = answer_generator.map(pprompts)
aresponses = answer_generator.map(aprompts)
tresponses = answer_generator.map(tprompts)

# Phase 3: assemble rows. a failed call keeps its row, with the negative class
# (False) and empty reasoning - only the failed metric is affected, the row's
# other metrics keep their real answers
rows = [
    [var1, var2,
     helpers.conclusion_of(presponse), helpers.reasoning_of(presponse),
     helpers.conclusion_of(aresponse), helpers.reasoning_of(aresponse),
     helpers.conclusion_of(tresponse), helpers.reasoning_of(tresponse),
     preport, areport, treport, label]
    for (var1, var2, label), presponse, aresponse, tresponse, preport, areport, treport
    in zip(pairs, presponses, aresponses, tresponses, preports, areports, treports)
]

columns = "Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Association", "Association Reasoning", "Temporality", "Temporality Reasoning", "Plausibility Report", "Association Report", "Temporality Report", "Label"
kgrag = pd.DataFrame(rows, columns=columns)
kgrag.to_csv(OUT_DIR / "kgrag.csv")

print(f"regenerated kgrag.csv with {len(kgrag)} predictions")
print(f"Plausibility F1: {f1_score(kgrag['Label'], kgrag['Plausibility'])}")

# %% [markdown]
# ## Step 2: consolidate undirected results (KG-RAG context only)

# %%
kgrag = kgrag.drop(columns=["Label"])
kgrag_cl = pd.read_csv(RESULT_DIR / "kg+rag_full_causal_literature.csv", index_col=0).drop(columns=["Label"])

kgrag_cl = consolidate_causal_literature(kgrag_cl)
undirected = pd.merge(kgrag, kgrag_cl, on=["Var1", "Var2"])
# same context tag wflow.py assigns - kept so this file's schema matches
# results_undirected_combined.csv restricted to the KG-RAG rows
undirected["context"] = "kg+llm_full.csv"

undirected.to_csv(OUT_DIR / "results_undirected_kgrag.csv")
print(f"consolidated {len(undirected)} undirected KG-RAG predictions")

# %% [markdown]
# ## Direction resolution
#
# Same logic as the two query_directions_*.py scripts, unified behind a
# use_proto flag and specialized to the KG-RAG context (tie-break contexts come
# from retrieve_kgrag_context, prompts from create_causal_prompt_kg).

# %%
direction_generator = get_client(schema=DirectionAnswer)

def resolve_bidirectional_edges(df, metric_col, metric_reasoning_col, metric_report_col, use_proto):
    """
    Resolve bidirectional edges for a given dataframe and metric, KG-RAG context.

    With use_proto=False this reproduces query_directions_without_proto.py: an
    edge is "present" in a direction iff the metric voted True. With
    use_proto=True it reproduces query_directions_with_proto.py: presence is the
    metric ORed with the proto model's edge, and absence requires both False.

    Returns a DataFrame with resolved directional edges and a Direction_Resolved
    flag (True = an LLM tie-break decided the direction).
    """

    # Self-merge to find pairs with both directions
    # this will drop pairs of variables that are constrained to one direction (Age, Sex, PEG)
    merged = df.merge(
        df,
        left_on=['Var1', 'Var2'],
        right_on=['Var2', 'Var1'],
        suffixes=('_fwd', '_rev')
    )

    # pairs present in only one direction never appear in the self-merge; they only
    # survive via the Sex/Age/PEG filters below - warn if any would be silently dropped
    all_pairs = set(zip(df['Var1'], df['Var2']))
    unrescued = [
        (v1, v2) for (v1, v2) in all_pairs
        if (v2, v1) not in all_pairs and v1 not in ("Sex", "Age") and v2 != "PEG"
    ]
    if unrescued:
        print(f"WARNING: {len(unrescued)} one-directional pair(s) not covered by the Sex/Age/PEG filters will be dropped: {sorted(unrescued)}")

    if use_proto:
        fwd_present = (merged[f'{metric_col}_fwd'] == True) | (merged['Proto_fwd'] == True)
        fwd_absent = (merged[f'{metric_col}_fwd'] == False) & (merged['Proto_fwd'] == False)
        rev_present = (merged[f'{metric_col}_rev'] == True) | (merged['Proto_rev'] == True)
        rev_absent = (merged[f'{metric_col}_rev'] == False) & (merged['Proto_rev'] == False)
        edge_mask = (df[metric_col] == True) | (df["Proto"] == True)
        proto_fwd_cols, proto_rev_cols = ['Proto_fwd'], ['Proto_rev']
    else:
        fwd_present = merged[f'{metric_col}_fwd'] == True
        fwd_absent = merged[f'{metric_col}_fwd'] == False
        rev_present = merged[f'{metric_col}_rev'] == True
        rev_absent = merged[f'{metric_col}_rev'] == False
        edge_mask = df[metric_col] == True
        proto_fwd_cols, proto_rev_cols = [], []

    # Filter for different scenarios
    fwd_true_rev_false = merged[fwd_present & rev_absent]
    fwd_false_rev_true = merged[fwd_absent & rev_present]
    both_true = merged[fwd_present & rev_present]

    # Clean up directional edges (already resolved)
    fwd_cols = ['Var1_fwd', 'Var2_fwd', f'{metric_col}_fwd', f'{metric_reasoning_col}_fwd', f'{metric_report_col}_fwd'] + proto_fwd_cols
    rev_cols = ['Var1_rev', 'Var2_rev', f'{metric_col}_rev', f'{metric_reasoning_col}_rev', f'{metric_report_col}_rev'] + proto_rev_cols

    fwd_true_rev_false_clean = fwd_true_rev_false[fwd_cols].rename(
        columns=lambda x: x.replace('_fwd', '')
    )
    fwd_false_rev_true_clean = fwd_false_rev_true[rev_cols].rename(
        columns=lambda x: x.replace('_rev', '')
    )

    # Get edges that are always directional (Sex, Age as Var1; PEG as Var2)
    sex_edges = df[(df["Var1"] == "Sex") & edge_mask]
    age_edges = df[(df["Var1"] == "Age") & edge_mask]
    peg_edges = df[(df["Var2"] == "PEG") & edge_mask]

    # Combine already directional edges
    directional_df = pd.concat(
        [fwd_true_rev_false_clean, fwd_false_rev_true_clean, sex_edges, age_edges, peg_edges],
        ignore_index=True
    ).drop_duplicates(subset=['Var1', 'Var2'])

    # mark which edges required resolution
    directional_df['Direction_Resolved'] = False

    # Get unique bidirectional pairs to resolve
    bidirectional_pairs = both_true.apply(
        # some extra logic to remove the duplicate pair in the other direction
        lambda x: tuple(sorted([x['Var1_fwd'], x['Var2_fwd']])), axis=1
    ).drop_duplicates().tolist()

    # Filter out pairs that are already resolved (Sex, Age, PEG)
    pairs_to_resolve = [
        pair for pair in bidirectional_pairs
        if 'Sex' not in pair and 'Age' not in pair and 'PEG' not in pair
    ]

    print(f"Found {len(pairs_to_resolve)} bidirectional pairs to resolve for {metric_col} using prompt")

    failed_pairs = []

    def _resolve_pair(pair):
        try:
            var1, var2 = pair
            prev_report = df[(df['Var1'] == var1) & (df['Var2'] == var2)][f'{metric_report_col}'].iloc[0]

            new_report = retrieve_kgrag_context(DIRECT_CAUSAL_PROMPT(var1, var2))
            response = direction_generator(create_causal_prompt_kg(var1, var2, new_report))

            return {
                'Var1': var1 if response.conclusion == 'A' else var2,
                'Var2': var2 if response.conclusion == 'A' else var1,
                metric_col: True,
                metric_reasoning_col: helpers.reasoning_to_string_multiple_choice(response),
                'Direction Report': new_report,
                metric_report_col: prev_report,
                'Direction_Resolved': True
            }

        except Exception as e:
            print(f"Error processing {pair}: {e}")
            failed_pairs.append(pair)
            # we return true here, because the knowledge system still voted for that direction, regardless
            # of whether it couldn't break the tie.
            # in practice, rarely errors.
            return {
                'Var1': pair[0],
                'Var2': pair[1],
                metric_col: True,
                metric_reasoning_col: str(e),
                'Direction Report': '',
                metric_report_col: '',
                'Direction_Resolved': True
            }

    # Query LLM for each pair concurrently
    results = thread_map(
        _resolve_pair, pairs_to_resolve, max_workers=config.LLM_MAX_WORKERS, desc=f"Resolving {metric_col}"
    )

    if failed_pairs:
        print(f"WARNING: {len(failed_pairs)}/{len(results)} direction resolutions failed for {metric_col}; "
              f"failed pairs emitted in arbitrary (alphabetical) direction with the error text as reasoning: {sorted(failed_pairs)}")

    # Combine with resolved directions
    if results:
        resolved_df = pd.DataFrame(results)
        final_df = pd.concat([directional_df, resolved_df], ignore_index=True).drop_duplicates(subset=['Var1', 'Var2'])
    else:
        final_df = directional_df

    return final_df

# %% [markdown]
# ## Step 3: resolve directions WITHOUT proto
#
# Causal literature is dropped entirely here: it stands alone and already
# accounts for direction (same as query_directions_without_proto.py).

# %%
# column masks, per metric (same as query_directions_without_proto.py)
plausibility_drop = ['Association', 'Association Reasoning', 'Temporality', 'Temporality Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning", "Causal Literature Report"]
association_drop = ['Plausibility', 'Plausibility Reasoning', 'Temporality', 'Temporality Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning", "Causal Literature Report"]
temporality_drop = ['Plausibility', 'Plausibility Reasoning', 'Association', 'Association Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning", "Causal Literature Report"]

print("Resolving directions WITHOUT proto...")
without_proto_dir = OUT_DIR / "directional_resolved_without_proto"
os.makedirs(without_proto_dir, exist_ok=True)

plausibility_resolved = resolve_bidirectional_edges(
    undirected.drop(columns=plausibility_drop), 'Plausibility', 'Plausibility Reasoning', 'Plausibility Report', use_proto=False
)
association_resolved = resolve_bidirectional_edges(
    undirected.drop(columns=association_drop), 'Association', 'Association Reasoning', 'Association Report', use_proto=False
)
temporality_resolved = resolve_bidirectional_edges(
    undirected.drop(columns=temporality_drop), 'Temporality', 'Temporality Reasoning', 'Temporality Report', use_proto=False
)

plausibility_resolved.to_csv(without_proto_dir / "kg_llm_plausibility_without_proto_resolved.csv", index=False)
association_resolved.to_csv(without_proto_dir / "kg_llm_association_without_proto_resolved.csv", index=False)
temporality_resolved.to_csv(without_proto_dir / "kg_llm_temporality_without_proto_resolved.csv", index=False)
print(f"saved to {without_proto_dir}/")

# %% [markdown]
# ## Step 4: resolve directions WITH proto
#
# Presence of an edge is the metric ORed with the proto model's edge, and causal
# literature IS resolved here, to compare it against the proto model (same as
# query_directions_with_proto.py).

# %%
proto = pd.read_csv(PROTO_DIR / "proto.csv")
proto["Proto"] = True
undirected_proto = undirected.merge(proto, how='left', on=['Var1', 'Var2'])
undirected_proto["Proto"] = undirected_proto["Proto"].fillna(False)

# column masks, per metric (same as query_directions_with_proto.py - the metric
# frames keep the causal literature report there, so they keep it here too)
plausibility_drop_p = ['Association', 'Association Reasoning', 'Temporality', 'Temporality Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning"]
association_drop_p = ['Plausibility', 'Plausibility Reasoning', 'Temporality', 'Temporality Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning"]
temporality_drop_p = ['Plausibility', 'Plausibility Reasoning', 'Association', 'Association Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning"]
causal_literature_drop_p = ['Plausibility', 'Plausibility Reasoning', 'Association', 'Association Reasoning', 'context', 'Temporality', 'Temporality Reasoning']

print("Resolving directions WITH proto...")
with_proto_dir = OUT_DIR / "directional_resolved_with_proto_and_causal_lit"
os.makedirs(with_proto_dir, exist_ok=True)

plausibility_proto_resolved = resolve_bidirectional_edges(
    undirected_proto.drop(columns=plausibility_drop_p), 'Plausibility', 'Plausibility Reasoning', 'Plausibility Report', use_proto=True
)
association_proto_resolved = resolve_bidirectional_edges(
    undirected_proto.drop(columns=association_drop_p), 'Association', 'Association Reasoning', 'Association Report', use_proto=True
)
temporality_proto_resolved = resolve_bidirectional_edges(
    undirected_proto.drop(columns=temporality_drop_p), 'Temporality', 'Temporality Reasoning', 'Temporality Report', use_proto=True
)
causal_lit_proto_resolved = resolve_bidirectional_edges(
    undirected_proto.drop(columns=causal_literature_drop_p), 'Causal Literature Prediction', 'Causal Literature Prediction Reasoning', 'Causal Literature Report', use_proto=True
)

plausibility_proto_resolved.to_csv(with_proto_dir / "kg_llm_plausibility_with_proto_resolved.csv", index=False)
association_proto_resolved.to_csv(with_proto_dir / "kg_llm_association_with_proto_resolved.csv", index=False)
temporality_proto_resolved.to_csv(with_proto_dir / "kg_llm_temporality_with_proto_resolved.csv", index=False)
causal_lit_proto_resolved.to_csv(with_proto_dir / "kg_llm_causal_lit_with_proto_resolved.csv", index=False)
print(f"saved to {with_proto_dir}/")

# %%
print("\nAll done. Files created:")
for root, _, files in sorted(os.walk(OUT_DIR)):
    for f in sorted(files):
        print(f"  {os.path.relpath(os.path.join(root, f), OUT_DIR)}")
