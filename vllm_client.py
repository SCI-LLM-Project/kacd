import requests
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

def render(messages: List[Dict[str, str]]) -> str:
    """Flatten chat-style messages into the single Mistral-formatted prompt string
    this server expects. Prompt-builder functions hand back plain messages so the
    same content works across backends; each backend renders it its own way."""
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_bos=True, add_generation_prompt=True
    )

class VLLMClient:
    # add json schema and use parameters to make it greedy
    def __init__(self, schema=None, url: str ="http://localhost:8000/generate") -> None:
        """
        Initialize the VLLM API wrapper.

        Args:
            url: Base URL of the VLLM server
        """
        self.url = url
        self.schema = schema

    def generate(self, messages: List[Dict[str, str]], sampling_params: Optional[Dict[Any, Any]]=None):
        """
        Generate text using VLLM with the specified sampling parameters.

        Args:
            messages: Chat-style messages (e.g. [{"role": "user", "content": ...}])
            sampling_params: sampling parameters as specified in the VLLM Sampling Params Object: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html

        Returns:
            Response from the VLLM server as a dictionary

        Raises:
            Exception: If the API request fails
        """
        prompt = render(messages)
        payload = {
            "prompt": prompt,
            "max_tokens": 10000  # Default max_tokens, can be overridden by sampling_params. 10000 chosen as an arbitrarily large number. Making this larger may make results slower.
        }

        # breaks down samping params and adds it to the api call.
        if sampling_params:
            for key, value in sampling_params.items():
                payload[key] = value
        
        if self.schema:
            payload["schema"] = self.schema.schema_json()
        
        response = requests.post(self.url, json=payload)
        try:
            # A common error is if the LLM loops through tokens until it hits the max limit, failing to output proper tokens for the schema.
            # Catch that error and regenerate the response using a repetition penalty to prevent looping
            unwrapped = self._unwrap(response, prompt)
        except Exception as e:
            print(f"Error in parsing LLM response: {e}")
            print(prompt)
            payload["repetition_penalty"] = 1.1
            try:
                response = requests.post(self.url, json=payload)
                unwrapped = self._unwrap(response, prompt)
            except Exception as e:
                print(e)
                print("Failed again with repetition penalty, returning empty list")
                return []

        return unwrapped

    def _unwrap(self, response, prompt):
        # takes out the prompt from the response, and converts it to the expected schema, if any
        if not self.schema:
            return response.json()["text"][0].split(prompt)[1]
            
        return self.schema.parse_raw(response.json()["text"][0].split(prompt)[1])
    
    def __call__(self, messages, **kwargs):
        return self.generate(messages, **kwargs)

    def map(self, list_of_messages, sampling_params=None):
        """Sequential fallback so callers can treat every backend uniformly - the
        local vLLM server is typically run with max-num-seqs=1, so concurrent
        dispatch wouldn't help here the way it does for a hosted API."""
        return [self(messages, sampling_params=sampling_params) for messages in list_of_messages]

