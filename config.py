NEO4J_URI = "bolt://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"
DIRECTORY = "../clbp_causal_md"

query_context_window = 8000
# topChunks, topCommunities, topRels is large so that we never not fill out the context window under any circumstance.
# doesn't affect performance because we already have filtering in place / limits on the context window.
topChunks = 50
topCommunities = 50
topRels = 50
# expanding this to 20 so that we can fill out the context window.
topEntities = 20

# LLM backend selection: "vllm" (local outlines.serve.serve server, default), or
# "together" (Together API, e.g. LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"
# or a fine-tuned model path like "your-org/your-model"). API keys are not stored
# here - see .env / LLM_API_KEY (read as TOGETHER_API_KEY-equivalent for "together").
LLM_BACKEND = "together"
LLM_MODEL = "damonlin93410-8d4a/test"
LLM_MAX_WORKERS = 32

# Tokenizer used for approximate token-count budgeting (context-window packing in
# build_context.py - query_context_window, community summarization, etc). Kept
# independent of LLM_MODEL on purpose: not every model name is loadable via
# AutoTokenizer.from_pretrained (a private/custom Together model like the one above
# has no public HF tokenizer at all). Point this at the closest public tokenizer for
# whatever model you're actually generating with, e.g. the base model a fine-tune
# was built on - it only needs to be roughly right, it's just used for size estimates.
TOKENIZER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"