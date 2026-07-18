NEO4J_URI = "bolt://localhost:7687"
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
# "openai_compatible" (any host speaking the OpenAI chat API - OpenAI, Azure OpenAI,
# Together, Fireworks, DeepInfra, Groq, or your own vLLM OpenAI-server, via LLM_BASE_URL).
# API keys are not stored here - see .env / LLM_API_KEY.
LLM_BACKEND = "vllm"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_BASE_URL = None
LLM_MAX_WORKERS = 8