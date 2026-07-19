"""
Golden-file tests for every pydantic schema actually passed as schema=... to
get_client()/APIClient() somewhere in the pipeline (see the SCHEMAS list
below - it was built by grepping the whole repo for `get_client(schema=`
and following each name back to its real class).

Each schema's exact response_format - the dict APIClient actually sends to
Together as response_format, built the same way APIClient.__init__ builds it
(schema.model_json_schema() run through the real _make_strict) - is captured
once as a fixture under tests/fixtures/response_formats/<Name>.json. The test
below rebuilds it fresh and diffs it against that fixture, so any change to a
model's fields, descriptions, or types (which changes what's actually sent to
the API) shows up as a failing test instead of silently drifting.

To intentionally update a fixture after a real schema change, regenerate it:
    python -m tests.test_api_client_schemas
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm.api_client import APIClient
from models.AnswerSchema import BooleanAnswer, CausalLitAnswer, DirectionAnswer
from models.ReportSchema import Report
from models.KnowledgeGraphSchema import KnowledgeGraph
from models.DisambiguateSchema import Disambiguate

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "response_formats"

# Every schema class actually used as get_client(schema=...) somewhere in the
# pipeline: query_kg.py/query_llm.py/query_rag.py -> BooleanAnswer,
# query_causal_lit_*.py -> CausalLitAnswer, query_directions_*.py ->
# DirectionAnswer, context_construction/build_context.py -> Report,
# construct_kg.py -> Disambiguate (via LLMGraphTransformer) and KnowledgeGraph
# (directly).
SCHEMAS = [BooleanAnswer, CausalLitAnswer, DirectionAnswer, Report, KnowledgeGraph, Disambiguate]


def _build_response_format(schema):
    """Builds the response_format exactly the way a real APIClient(schema=...)
    would, with Together itself mocked out so this never touches the network."""
    with patch("llm.api_client.Together") as fake_cls:
        fake_cls.return_value = MagicMock()
        client = APIClient(schema=schema, model="fixture-generation")
    return client._response_format


def _fixture_path(schema):
    return FIXTURES_DIR / f"{schema.__name__}.json"


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_response_format_matches_fixture(schema):
    fixture_path = _fixture_path(schema)
    assert fixture_path.exists(), (
        f"no fixture for {schema.__name__} at {fixture_path} - "
        f"run `python -m tests.test_api_client_schemas` to generate it"
    )

    expected = json.loads(fixture_path.read_text())
    actual = _build_response_format(schema)

    assert actual == expected, (
        f"{schema.__name__}'s response_format no longer matches "
        f"tests/fixtures/response_formats/{schema.__name__}.json - if this schema "
        f"change to the model was intentional, regenerate the fixture with "
        f"`python -m tests.test_api_client_schemas`"
    )


def test_every_production_schema_has_a_fixture():
    """Guards against SCHEMAS silently falling out of sync with reality -
    fails loudly if get_client(schema=...) picks up a class not listed above."""
    fixture_names = {p.stem for p in FIXTURES_DIR.glob("*.json")}
    schema_names = {s.__name__ for s in SCHEMAS}
    assert fixture_names == schema_names


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for schema in SCHEMAS:
        response_format = _build_response_format(schema)
        _fixture_path(schema).write_text(json.dumps(response_format, indent=2, sort_keys=True) + "\n")
        print(f"wrote {_fixture_path(schema)}")
