import requests
import json
import msgspec
from vllm import SamplingParams
from typing import Optional, Dict, Any

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
        
    def generate(self, prompt: str, sampling_params: Optional[Dict[Any, Any]]=None):
        """
        Generate text using VLLM with the specified sampling parameters.
        
        Args:
            prompt: Input prompt text
            sampling_params: sampling parameters as specified in the VLLM Sampling Params Object: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        
        Returns:
            Response from the VLLM server as a dictionary
            
        Raises:
            Exception: If the API request fails
        """
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
    
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

