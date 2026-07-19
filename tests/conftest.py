"""
context_construction/build_context.py executes real, unguarded side-effecting
calls at MODULE IMPORT time (not inside any function) - it builds a real
Neo4jGraph connection and a real APIClient/Together client as soon as it's
imported. Ordinary per-test monkeypatching can't help with that: the module
has to import successfully first, and by then those real calls already fired.

So the first import of context_construction.build_context anywhere in this
test suite has to happen here, guarded, exactly once. Python then caches the
module in sys.modules, so every later `import context_construction.build_context`
(this file or any future test file) reuses this already-safely-constructed
module object without re-running its top-level code.

Any new test file that needs context_construction.build_context should just
`import context_construction.build_context as build_context` normally - no
need to repeat this guard.
"""
import os
from unittest.mock import MagicMock, patch

# util.helpers (transitively imported by build_context) loads a real
# HuggingFace tokenizer at import time. The model is already cached locally,
# but AutoTokenizer.from_pretrained still attempts a network freshness check
# by default even on a cache hit - force it to skip that and resolve straight
# from cache. setdefault (not a hard overwrite) so a developer who genuinely
# wants a fresh pull can still override via their shell env.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

with patch("langchain_community.graphs.Neo4jGraph") as fake_neo4j_graph_cls, \
     patch("llm.api_client.Together") as fake_together_cls:
    fake_neo4j_graph_cls.return_value = MagicMock(name="Neo4jGraph")
    fake_together_cls.return_value = MagicMock(name="Together")
    import context_construction.build_context  # noqa: F401
