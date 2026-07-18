from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from openai import OpenAI


class APIClient:
    """
    Drop-in replacement for VLLMClient backed by any OpenAI-compatible chat API
    (OpenAI, Azure OpenAI, Together, Fireworks, DeepInfra, Groq, a vLLM
    OpenAI-server, ...), selected via base_url.
    """

    def __init__(
        self,
        schema=None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        max_workers: int = 8,
    ) -> None:
        self.schema = schema
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.max_workers = max_workers

    def __call__(self, messages: List[Dict[str, str]], sampling_params: Optional[Dict[str, Any]] = None):
        sampling_params = sampling_params or {}
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=sampling_params.get("temperature", 0),
        )
        if self.schema:
            patched = _make_strict(self.schema.model_json_schema())
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema.__name__,
                    "strict": True,
                    "schema": patched,
                },
            }

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
        """Run __call__ over many independent prompts concurrently, preserving order."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self, messages, sampling_params=sampling_params) for messages in list_of_messages]
            return [f.result() for f in futures]


def _make_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively force additionalProperties=false and a full `required` list onto
    every object node in the schema, including everything under $defs.
    Outlines' own OpenAI integration only patches the top-level schema, which
    OpenAI's real strict mode rejects for any schema with nested models - every
    schema in this repo (KnowledgeGraph, Disambiguate, Report, Answer) has nested
    BaseModel fields, so this has to walk the whole tree.
    """
    for node in [schema, *schema.get("$defs", {}).values()]:
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}).keys())
    return schema
