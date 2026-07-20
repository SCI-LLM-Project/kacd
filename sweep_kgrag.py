# %% [markdown]
# # KG+RAG hyperparameter sweep
#
# Sweeps the retrieval / context-packing hyperparameters of the KG+RAG method
# (query_kg.py + context_construction/retriever_kgrag.py) and scores each
# configuration against the labels in data/full_cleaned.csv.
#
# Hyperparameters swept (baseline = current config.py values):
#   top_entities    - k for the vector similarity search (config.topEntities)
#   chunk_frac      - share of the token budget reserved for chunks (hardcoded 0.5
#                     in construct_query_context)
#   report_frac     - share reserved for community reports; relationships get the
#                     leftovers (hardcoded 0.4)
#
# The retrieval caps (topChunks/topCommunities/topRels, 50 each) are NOT swept:
# the 8000-token budget is the binding constraint, so the packed context stops
# well before 50 items of anything - per config.py's own comment, the caps only
# exist so the window always fills. They stay at their config values.
#
# The total token budget for the packed report is held fixed at 8000 tokens
# (config.query_context_window) - only the caps and the split of that budget are
# swept. The LLM itself (config.LLM_MODEL etc.) is also held fixed.
#
# Cost controls:
#   - neo4j retrieval cached per (query, top_entities); maximal lists are fetched
#     once and smaller caps applied by slicing (equivalent - the cypher returns
#     each list already sorted)
#   - LLM responses cached on disk keyed by a hash of the full prompt, so identical
#     prompts across configs / re-runs are never paid for twice
#   - stratified subsample of the 145 pairs (SAMPLE_N), and a one-factor-at-a-time
#     sweep instead of a full cartesian grid
#   - results checkpointed to CSV after every config
#
# Prerequisites: neo4j running in docker, LLM_API_KEY in .env, run from the
# project root:  python sweep_kgrag.py
# (or in tmux like wflow.py; progress goes to stdout)

# %%
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import hashlib
import itertools
import json
import pickle
import time

import matplotlib

matplotlib.use("Agg")  # headless - the plot is saved to a file, never shown
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import config
import util.helpers as helpers
from llm.factory import get_client
from models.AnswerSchema import BooleanAnswer as Answer
from prompts.query_prompts.base_prompts import query_kg_prompt
from prompts.query_prompts.metric_prompts import (
    association_prompt,
    plausibility_prompt,
    temporality_prompt,
)

# Reuse the shared retriever setup (neo4j driver, vector index, embedding model,
# Neo4jVector) rather than duplicating it. lc_vector takes k and the top-N caps
# per call, so nothing needs re-configuring to sweep them.
from context_construction.retriever_kgrag import lc_vector
from context_construction.build_context import stringify_report, format_triplet

# %% [markdown]
# ## Settings

# %%
# Each config costs len(pairs) * len(ACTIVE_METRICS) LLM calls before cache hits.
# Sweep on a subsample first, then re-run the best configs with SAMPLE_N = None
# to confirm on all 145 pairs.
SAMPLE_N = 60  # None -> use all 145 pairs
SEED = 0

# query_kg.py scores Plausibility, so that's the default. Add the others to
# sweep them too (each one multiplies the LLM calls).
ACTIVE_METRICS = ["Plausibility"]

# The values the pipeline runs with today - every sweep config is measured
# against this.
BASELINE = dict(
    top_entities=config.topEntities,
    top_chunks=config.topChunks,
    top_communities=config.topCommunities,
    top_rels=config.topRels,
    context_window=config.query_context_window,  # fixed at 8000 - not swept
    chunk_frac=0.5,
    report_frac=0.4,
)

# One-factor-at-a-time grid: every config is the baseline with a single
# hyperparameter changed. chunk_frac + report_frac must stay <= 1.0 against the
# baseline value of the other fraction (0.5/0.4), hence the 0.6/0.5 caps.
GRID = {
    "chunk_frac": [0.3, 0.5, 0.6],
    "report_frac": [0.2, 0.4, 0.5],
    "top_entities": [5, 10, 20, 40],
}

RETRIEVAL_CAP = 50  # what the cypher is asked for; the baseline caps then slice it

METRICS = {
    "Plausibility": plausibility_prompt,
    "Association": association_prompt,
    "Temporality": temporality_prompt,
}

SWEEP_DIR = config.RESULT_DIR / "kgrag_sweeps"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Parameterized retrieval + context packing
#
# retrieve_raw does the neo4j round trip; pack_context turns a raw retrieval into
# the report string the prompt sees. Splitting them means one retrieval serves
# every config that shares the same top_entities - the caps and the token-budget
# split are applied after the fact.

# %%
# Retrieval always asks the cypher for RETRIEVAL_CAP chunks/communities/rels and
# caches per (query, top_entities). Each list comes back sorted by the same
# ordering a smaller LIMIT would use, so slicing to a smaller top-N later is
# equivalent to having retrieved with it. Cache is in-memory only, so a rebuilt
# KG just needs a fresh process to invalidate it.
_retrieval_cache = {}

def retrieve_raw(query, top_entities):
    key = (query, top_entities)
    if key not in _retrieval_cache:
        res = lc_vector.similarity_search(
            query,
            k=top_entities,
            params={
                "topChunks": RETRIEVAL_CAP,
                "topCommunities": RETRIEVAL_CAP,
                "topRels": RETRIEVAL_CAP,
            },
        )
        metadata = res[0].metadata
        _retrieval_cache[key] = {
            "reports": [stringify_report(report) for report in metadata["Reports"]],
            "chunks": metadata["Chunks"],
            "relationships": [format_triplet(triplet) for triplet in metadata["Relationships"]],
        }
    return _retrieval_cache[key]

def pack_context(raw, p):
    """Parameterized mirror of build_context.construct_query_context: same packing
    order (reports, then chunks, then relationships take whatever budget is left)
    and same output format, but the top-N caps and the budget split are
    hyperparameters instead of config constants."""
    reports = raw["reports"][: p["top_communities"]]
    chunks = raw["chunks"][: p["top_chunks"]]
    relationships = raw["relationships"][: p["top_rels"]]

    max_report_tokens = p["report_frac"] * p["context_window"]
    max_chunk_tokens = p["chunk_frac"] * p["context_window"]

    top_reports, total_report_count = [], 0
    for report in reports:
        count = helpers.token_count(report)
        if count + total_report_count > max_report_tokens:
            break
        top_reports.append(report)
        total_report_count += count

    top_chunks, total_chunk_count = [], 0
    for chunk in chunks:
        count = helpers.token_count(chunk)
        if count + total_chunk_count > max_chunk_tokens:
            break
        top_chunks.append(chunk)
        total_chunk_count += count

    remaining = p["context_window"] - total_report_count - total_chunk_count
    top_triplets, total_triplet_count = [], 0
    for triplet in relationships:
        count = helpers.token_count(triplet)
        if total_triplet_count + count > remaining:
            break
        top_triplets.append(triplet)
        total_triplet_count += count

    return (
        "# Report:\n"
        "## Chunks:\n\n" +
        "\n".join(top_chunks) +
        "## Communities:\n\n" +
        "\n".join(top_reports) +
        "\n## Relationships:\n\n" +
        "".join(top_triplets)
    )

# %% [markdown]
# ## Evaluation data

# %%
full = pd.read_csv(f"{config.PROJECT_ROOT}/data/full_cleaned.csv").drop(columns=["Unnamed: 0"])

# same bandaid as query_kg.py - the KG entity is "Sleep disturbance", not "Sleep"
def fix(var):
    return "Sleep disturbance" if var == "Sleep" else var

eval_df = full
if SAMPLE_N is not None:
    eval_df = pd.concat([
        group.sample(n=max(1, round(SAMPLE_N * len(group) / len(full))), random_state=SEED)
        for _, group in full.groupby("label")
    ]).sort_index()

pairs = [(fix(row["var1"]), fix(row["var2"]), row["label"]) for _, row in eval_df.iterrows()]
print(f"{len(pairs)} pairs ({int(eval_df['label'].sum())} positive / {int((~eval_df['label']).sum())} negative)")

# %% [markdown]
# ## LLM client + prompt-level response cache

# %%
generator = get_client(schema=Answer)

# Disk-backed response cache keyed by a hash of (model, full prompt). Any config
# or re-run that produces a prompt already seen gets the stored answer for free;
# only genuinely new prompts hit the API.
LLM_CACHE_PATH = SWEEP_DIR / "llm_cache.pkl"
try:
    llm_cache = pickle.loads(LLM_CACHE_PATH.read_bytes()) if LLM_CACHE_PATH.exists() else {}
except Exception:
    print("WARNING: could not load the LLM cache (schema changed?) - starting fresh")
    llm_cache = {}
print(f"{len(llm_cache)} cached LLM responses")

def prompt_key(messages):
    payload = json.dumps([config.LLM_MODEL, messages], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def generate_cached(prompts):
    keys = [prompt_key(prompt) for prompt in prompts]
    todo = [(key, prompt) for key, prompt in zip(keys, prompts) if key not in llm_cache]
    if todo:
        responses = generator.map([prompt for _, prompt in todo])
        for (key, _), response in zip(todo, responses):
            if response is not None:  # failed calls stay uncached so a re-run retries them
                llm_cache[key] = response
        LLM_CACHE_PATH.write_bytes(pickle.dumps(llm_cache))
    return [llm_cache.get(key) for key in keys]

# %% [markdown]
# ## Run one configuration

# %%
def run_config(p, pairs):
    """Retrieve, pack, query, and score one hyperparameter configuration.
    Returns a flat result row plus the per-pair predictions."""
    assert p["chunk_frac"] + p["report_frac"] <= 1.0, "budget fractions exceed the context window"
    row = dict(p)
    labels = [label for _, _, label in pairs]
    predictions = {}

    for metric in ACTIVE_METRICS:
        queries = [METRICS[metric](var1, var2) for var1, var2, _ in pairs]
        raws = [retrieve_raw(query, p["top_entities"])
                for query in tqdm(queries, desc=f"{metric} retrieval", leave=False)]
        contexts = [pack_context(raw, p) for raw in raws]
        prompts = [
            query_kg_prompt(query, var1, var2, context, config.def_map)
            for (var1, var2, _), query, context in zip(pairs, queries, contexts)
        ]
        responses = generate_cached(prompts)
        preds = [helpers.conclusion_of(response) for response in responses]
        predictions[metric] = preds

        row[f"{metric}_f1"] = f1_score(labels, preds)
        row[f"{metric}_precision"] = precision_score(labels, preds, zero_division=0)
        row[f"{metric}_recall"] = recall_score(labels, preds, zero_division=0)
        row[f"{metric}_accuracy"] = accuracy_score(labels, preds)
        row[f"{metric}_failed_calls"] = sum(response is None for response in responses)
        row[f"{metric}_mean_context_tokens"] = sum(helpers.token_count(c) for c in contexts) / len(contexts)

    row["n_pairs"] = len(pairs)
    return row, predictions

# %% [markdown]
# ## Build the sweep configs

# %%
def one_factor_configs(grid, baseline):
    """Baseline plus every config that changes exactly one hyperparameter."""
    configs = [{**baseline, "varied": "baseline"}]
    for name, values in grid.items():
        for value in values:
            if value == baseline[name]:
                continue
            configs.append({**baseline, name: value, "varied": name})
    return configs

def full_grid_configs(grid, baseline):
    """Full cartesian product - use for a targeted refinement grid, not GRID above."""
    names = list(grid)
    configs = []
    for values in itertools.product(*grid.values()):
        c = {**baseline, **dict(zip(names, values))}
        if c["chunk_frac"] + c["report_frac"] > 1.0:
            continue
        c["varied"] = "grid"
        configs.append(c)
    return configs

configs = one_factor_configs(GRID, BASELINE)

upper_bound = len(configs) * len(pairs) * len(ACTIVE_METRICS)
print(f"{len(configs)} configs -> at most {upper_bound} LLM calls (cache hits are free)")

# %% [markdown]
# ## Run the sweep
#
# Checkpoints to CSV after every config, so an interrupted sweep loses at most one
# config - and thanks to the LLM cache, re-running it is nearly free up to where
# it stopped.

# %%
run_stamp = time.strftime("%Y%m%d_%H%M%S")
SWEEP_PATH = SWEEP_DIR / f"sweep_{run_stamp}.csv"

rows = []
for p in tqdm(configs, desc="configs"):
    params = {key: value for key, value in p.items() if key != "varied"}
    row, _ = run_config(params, pairs)
    row["varied"] = p["varied"]
    rows.append(row)
    pd.DataFrame(rows).to_csv(SWEEP_PATH, index=False)  # checkpoint after every config

results = pd.DataFrame(rows)
print(f"saved to {SWEEP_PATH}")

# %% [markdown]
# ## Results

# %%
score_col = f"{ACTIVE_METRICS[0]}_f1"
summary = results.sort_values(score_col, ascending=False)[
    ["varied", *GRID,
     score_col,
     f"{ACTIVE_METRICS[0]}_precision",
     f"{ACTIVE_METRICS[0]}_recall",
     f"{ACTIVE_METRICS[0]}_mean_context_tokens",
     f"{ACTIVE_METRICS[0]}_failed_calls"]
]
print(summary.head(15).to_string(index=False))

# %%
SERIES = "#2a78d6"
MUTED = "#767676"

baseline_score = results.loc[results["varied"] == "baseline", score_col].iloc[0]
params_to_plot = [name for name in GRID if (results["varied"] == name).any()]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, name in zip(axes.flat, params_to_plot):
    # the baseline row supplies the point at the baseline value of this parameter
    sub = results[results["varied"].isin([name, "baseline"])].sort_values(name)
    ax.plot(sub[name], sub[score_col], marker="o", color=SERIES, linewidth=2, markersize=6)
    ax.axhline(baseline_score, color=MUTED, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_title(name, fontsize=10)
    ax.grid(alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
for ax in axes.flat[len(params_to_plot):]:
    ax.axis("off")
axes[0].set_ylabel(score_col)
fig.suptitle(f"{score_col} vs each hyperparameter (dashed = baseline)", y=1.02)
fig.tight_layout()
PLOT_PATH = SWEEP_DIR / f"sweep_{run_stamp}.png"
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"plot saved to {PLOT_PATH}")

# %% [markdown]
# ## Inspecting a configuration qualitatively
#
# Numbers say which config wins; reading its packed context says why. Call this
# from a REPL / interactive session after (or instead of) a sweep - retrieval is
# cached, so it's instant for already-swept configs.

# %%
def inspect_context(p=None, pair_index=0, metric=None, max_chars=4000):
    p = p or BASELINE
    metric = metric or ACTIVE_METRICS[0]
    var1, var2, label = pairs[pair_index]
    query = METRICS[metric](var1, var2)
    context = pack_context(retrieve_raw(query, p["top_entities"]), p)
    print(f"{var1} / {var2} (label={label}) - {helpers.token_count(context)} tokens\n")
    print(context[:max_chars])

# %% [markdown]
# ## Making a winner permanent
#
# Once a configuration holds up on the full 145 pairs (SAMPLE_N = None), wire it
# into the pipeline:
#   - top_entities -> topEntities in config.py
#   - chunk_frac / report_frac -> the two hardcoded fractions in
#     construct_query_context (context_construction/build_context.py)
# then re-run query_kg.py (ideally under a new RUN_NAME) to regenerate kgrag.csv.
