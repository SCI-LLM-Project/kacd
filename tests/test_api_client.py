from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm.api_client import APIClient, _make_strict


class Inner(BaseModel):
    value: int


class Outer(BaseModel):
    name: str
    inner: Inner


def _fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestMakeStrict:
    def test_marks_top_level_object_strict(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        result = _make_strict(schema)
        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"a", "b"}

    def test_marks_nested_defs_strict_too(self):
        schema = Outer.model_json_schema()
        result = _make_strict(schema)

        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"name", "inner"}

        inner_def = result["$defs"]["Inner"]
        assert inner_def["additionalProperties"] is False
        assert set(inner_def["required"]) == {"value"}

    def test_leaves_non_object_nodes_alone(self):
        schema = {"type": "string"}
        result = _make_strict(schema)
        assert "additionalProperties" not in result
        assert "required" not in result

    def test_empty_object_gets_empty_required_list(self):
        schema = {"type": "object", "properties": {}}
        result = _make_strict(schema)
        assert result["required"] == []


@pytest.fixture
def fake_together():
    """Patches llm.api_client.Together so APIClient() never touches the network.
    Yields the fake client instance; tests configure its
    .chat.completions.create mock directly."""
    with patch("llm.api_client.Together") as fake_cls:
        instance = MagicMock()
        fake_cls.return_value = instance
        yield instance


class TestAPIClientInit:
    def test_no_schema_means_no_response_format(self, fake_together):
        client = APIClient(schema=None, model="m")
        assert client._response_format is None

    def test_schema_builds_strict_response_format_with_schema_name(self, fake_together):
        client = APIClient(schema=Outer, model="m")
        assert client._response_format["type"] == "json_schema"
        assert client._response_format["json_schema"]["name"] == "Outer"
        assert client._response_format["json_schema"]["schema"]["additionalProperties"] is False


class TestAPIClientCall:
    def test_parses_response_into_schema(self, fake_together):
        fake_together.chat.completions.create.return_value = _fake_response('{"name": "x", "inner": {"value": 5}}')
        client = APIClient(schema=Outer, model="m")

        result = client([{"role": "user", "content": "hi"}])

        assert result == Outer(name="x", inner=Inner(value=5))

    def test_no_schema_returns_raw_text(self, fake_together):
        fake_together.chat.completions.create.return_value = _fake_response("raw text")
        client = APIClient(schema=None, model="m")

        result = client([{"role": "user", "content": "hi"}])

        assert result == "raw text"

    def test_passes_model_messages_and_defaults_into_create(self, fake_together):
        fake_together.chat.completions.create.return_value = _fake_response('{"name": "x", "inner": {"value": 1}}')
        client = APIClient(schema=Outer, model="my-model", max_tokens=123)

        messages = [{"role": "user", "content": "hi"}]
        client(messages)

        _, kwargs = fake_together.chat.completions.create.call_args
        assert kwargs["model"] == "my-model"
        assert kwargs["messages"] == messages
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 123
        assert kwargs["response_format"] == client._response_format

    def test_sampling_params_override_defaults(self, fake_together):
        fake_together.chat.completions.create.return_value = _fake_response('{"name": "x", "inner": {"value": 1}}')
        client = APIClient(schema=Outer, model="m", max_tokens=999)

        client([{"role": "user", "content": "hi"}], sampling_params={"temperature": 0.7, "max_tokens": 55})

        _, kwargs = fake_together.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 55

    def test_retries_once_on_failure_then_succeeds(self, fake_together):
        fake_together.chat.completions.create.side_effect = [
            Exception("transient"),
            _fake_response('{"name": "x", "inner": {"value": 1}}'),
        ]
        client = APIClient(schema=Outer, model="m")

        result = client([{"role": "user", "content": "hi"}])

        assert result == Outer(name="x", inner=Inner(value=1))
        assert fake_together.chat.completions.create.call_count == 2

    def test_second_failure_propagates_uncaught(self, fake_together):
        fake_together.chat.completions.create.side_effect = [
            Exception("first failure"),
            Exception("second failure"),
        ]
        client = APIClient(schema=Outer, model="m")

        with pytest.raises(Exception, match="second failure"):
            client([{"role": "user", "content": "hi"}])

        assert fake_together.chat.completions.create.call_count == 2


class TestAPIClientMap:
    def test_preserves_order_and_isolates_per_item_failures(self, fake_together):
        # "fail" raises on *every* attempt (both the initial try and the retry
        # inside __call__), so __call__ re-raises and map() must catch only
        # that item, leaving the others - and their order - intact.
        def create(**kwargs):
            content = kwargs["messages"][0]["content"]
            if content == "fail":
                raise Exception(f"boom for {content}")
            return _fake_response(f'{{"name": "{content}", "inner": {{"value": 1}}}}')

        fake_together.chat.completions.create.side_effect = create
        client = APIClient(schema=Outer, model="m", max_workers=4)

        list_of_messages = [
            [{"role": "user", "content": "ok0"}],
            [{"role": "user", "content": "fail"}],
            [{"role": "user", "content": "ok2"}],
        ]

        results = client.map(list_of_messages)

        assert results[0] == Outer(name="ok0", inner=Inner(value=1))
        assert results[1] is None
        assert results[2] == Outer(name="ok2", inner=Inner(value=1))
