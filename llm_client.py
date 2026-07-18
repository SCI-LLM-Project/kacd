from dotenv import dotenv_values

import config
from api_client import APIClient
from vllm_client import VLLMClient


def get_client(schema=None):
    """
    Single choke point for picking the active LLM backend. Swapping models/hosts
    is a config.py edit (plus LLM_API_KEY in .env for hosted APIs), not a
    per-call-site code change.
    """
    if config.LLM_BACKEND == "vllm":
        return VLLMClient(schema=schema)

    if config.LLM_BACKEND == "together":
        api_key = dotenv_values(dotenv_path=".env").get("LLM_API_KEY")
        return APIClient(
            schema=schema,
            model=config.LLM_MODEL,
            api_key=api_key,
            max_workers=config.LLM_MAX_WORKERS,
        )

    raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}")
