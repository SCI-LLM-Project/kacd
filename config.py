import json
from pathlib import Path

# anchored to this file's own location rather than a hardcoded path, so every
# path below resolves the same regardless of the working directory a script or
# notebook is launched from
PROJECT_ROOT = Path(__file__).resolve().parent

# base of the results tree - write through RESULT_DIR below, not this
RESULTS_ROOT = PROJECT_ROOT / "results"
# name of the execution run. Everything the pipeline generates lands in
# results/<RUN_NAME>/, so changing this before a run (different model, prompt
# edits, a rebuilt KG) keeps runs side by side instead of overwriting.
RUN_NAME = "llama"
RESULT_DIR = RESULTS_ROOT / RUN_NAME
# tracked pipeline inputs (proto edges are external and never regenerated here)
DATA_DIR = PROJECT_ROOT / "data"
PROTO_DIR = DATA_DIR / "proto"

# variables of interest - shared definitions used across query prompts
with open(PROJECT_ROOT / "variable_definitions/default_definitions.json", "r") as file:
    def_map = json.load(file)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"
DIRECTORY = "../clbp_causal_md_24jul26"

query_context_window = 8000
# We chose these parameters empirically picking numbers high enough that the KG-RAG context will always be close to 8000
# topChunks, topCommunities, topRels is large so that we never not fill out the context window under any circumstance.
# doesn't affect performance because we already have filtering in place / limits on the context window.
topChunks = 50
topCommunities = 50
topRels = 50
# expanding this to 20 so that we can fill out the context window.
topEntities = 20

# LLM backend selection - Together API (LLM_MODEL, e.g.
# "meta-llama/Llama-3.3-70B-Instruct-Turbo", or a fine-tuned model's dedicated
# endpoint name). API key is not stored here - see .env / LLM_API_KEY.
LLM_BACKEND = "together"
LLM_MODEL = "together_endpoint_name"
LLM_MAX_WORKERS = 24
# Max tokens the model is allowed to generate per response (output only - has
# no bearing on prompt/input size). Applied system-wide via get_client(); every
# call site already relied on this same value as APIClient's own default, this
# just makes it an explicit, centrally-configured setting instead of an
# implicit one.
LLM_MAX_TOKENS = 10000

# Tokenizer used for approximate token-count budgeting (context-window packing in
# build_context.py - query_context_window, community summarization, etc). Kept
# independent of LLM_MODEL on purpose: not every model name is loadable via
# AutoTokenizer.from_pretrained (a private/custom Together model like the one above
# has no public HF tokenizer at all). Point this at the closest public tokenizer for
# whatever model you're actually generating with, e.g. the base model a fine-tune
# was built on - it only needs to be roughly right, it's just used for size estimates.
TOKENIZER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
