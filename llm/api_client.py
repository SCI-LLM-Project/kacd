from typing import Any, Dict, List, Optional

from tqdm.contrib.concurrent import thread_map
from together import Together


class APIClient:
    """
    LLM client backed by the Together API (client.chat.completions.create,
    same response shape as openai-python), constrained to a Pydantic schema
    via response_format.
    """

    def __init__(
        self,
        schema=None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        max_workers: int = 8,
        max_tokens: int = 10000,
    ) -> None:
        self.schema = schema
        self.model = model
        # api_key=None falls back to the TOGETHER_API_KEY env var, same as the SDK's own default
        self.client = Together(api_key=api_key, max_retries=max_retries)
        self.max_workers = max_workers
        # an arbitrarily large cap, not a target. without this, a truncated response
        # fails model_validate_json and burns a full retry round-trip for something
        # that was never going to parse either time.
        self.max_tokens = max_tokens

        # schema never changes for the life of this client - patch and build the
        # response_format once here instead of redoing it on every __call__
        self._response_format = None
        if schema:
            patched = _make_strict(schema.model_json_schema())
            self._response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": patched,
                },
            }

    def __call__(self, messages: List[Dict[str, str]], sampling_params: Optional[Dict[str, Any]] = None):
        sampling_params = sampling_params or {}
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=sampling_params.get("temperature", 0),
            max_tokens=sampling_params.get("max_tokens", self.max_tokens),
        )
        if self._response_format:
            kwargs["response_format"] = self._response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            return self.schema.model_validate_json(text) if self.schema else text
        except Exception as e:
            print(f"Error in APIClient call, retrying once: {e}")
            response = self.client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            return self.schema.model_validate_json(text) if self.schema else text

    def map(self, list_of_messages: List[List[Dict[str, str]]], sampling_params: Optional[Dict[str, Any]] = None):
        """Run __call__ over many independent prompts concurrently, preserving order,
        with a progress bar via tqdm's own ThreadPoolExecutor integration. thread_map
        itself has no per-item exception isolation - a raised exception would blow
        away every result, not just the failing one - so the try/except has to live
        inside the function it calls, not around thread_map."""
        def _safe_call(indexed_messages):
            i, messages = indexed_messages
            try:
                return self(messages, sampling_params=sampling_params)
            except Exception as e:
                print(f"Error in APIClient.map() item {i}, skipping: {e}")
                return None

        return thread_map(
            _safe_call,
            list(enumerate(list_of_messages)),
            max_workers=self.max_workers,
            desc="APIClient.map",
        )


def _make_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively force additionalProperties=false and a full `required` list onto
    every object node in the schema, including everything under $defs. Together's
    grammar-constrained decoding reportedly handles nested schemas fine without
    this (unlike OpenAI's strict mode, which rejects nested models without it),
    but it's cheap and only tightens the constraint - it stops the model from
    slipping in extra keys - so it's applied regardless of provider.
    """
    for node in [schema, *schema.get("$defs", {}).values()]:
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}).keys())
    return schema
