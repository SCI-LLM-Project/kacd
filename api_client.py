from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from together import Together


class APIClient:
    """
    Drop-in replacement for VLLMClient backed by the Together API
    (client.chat.completions.create, same response shape as openai-python).
    """

    def __init__(
        self,
        schema=None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        max_workers: int = 8,
    ) -> None:
        self.schema = schema
        self.model = model
        # api_key=None falls back to the TOGETHER_API_KEY env var, same as the SDK's own default
        self.client = Together(api_key=api_key, max_retries=max_retries)
        self.max_workers = max_workers

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
        """Run __call__ over many independent prompts concurrently, preserving order.
        A single item's failure doesn't abort the batch - its slot in the returned
        list is None (logged), same convention as VLLMClient's existing [] fallback."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self, messages, sampling_params=sampling_params) for messages in list_of_messages]
            results = []
            for i, f in enumerate(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"Error in APIClient.map() item {i}, skipping: {e}")
                    results.append(None)
            return results


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
