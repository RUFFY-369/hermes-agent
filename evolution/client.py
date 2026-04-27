import asyncio
from typing import List, Any, Optional, Dict
import re
import os
import httpx
from pydantic import BaseModel

class RolloutResponse(BaseModel):
    text: str = ""
    raw_text: str = ""
    logprobs: List[Any] = []

class SGLangClient:
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self._session = None

    async def get_session(self):
        if self._session is None:
            self._session = httpx.AsyncClient(timeout=600.0)
        return self._session

    async def generate(self, prompt: str, **kwargs) -> RolloutResponse:
        session = await self.get_session()
        try:
            # Use the exact same structure that worked in curl
            payload = {
                "text": prompt,
                "max_new_tokens": 1024,
                "temperature": 0.8,
                "stop": ["```"],
                "return_logprob": True
            }
            
            # Check for LoRA path and include it if valid
            lora_path = kwargs.get("lora_path")
            if lora_path and os.path.exists(lora_path):
                payload["lora_path"] = lora_path
            
            # Add any other kwargs
            for k, v in kwargs.items():
                if k != "lora_path":
                    payload[k] = v

            response = await session.post(f"{self.base_url}/generate", json=payload)
            if response.status_code != 200:
                # print(f"SGLang Error: {response.text}")
                return RolloutResponse(text="")
                
            data = response.json()
            raw_gen = data.get("text", "")
            
            extracted = raw_gen
            
            # Prepend header if missing
            if "def solve():" in prompt and "def solve():" not in extracted:
                extracted = "def solve():\n" + extracted

            # Append call if missing
            if "def solve" in extracted and "print(solve())" not in extracted:
                extracted += "\n\nprint(solve())"
            
            # Extract logprobs from meta_info -> output_token_logprobs
            meta = data.get("meta_info", {})
            logprobs = meta.get("output_token_logprobs", [])
            
            return RolloutResponse(text=extracted, raw_text=raw_gen, logprobs=logprobs)
        except Exception as e:
            # print(f"Generate Error: {e}")
            return RolloutResponse(text="")

    async def generate_group(self, prompts, **kwargs):
        semaphore = asyncio.Semaphore(128) # Flood SGLang for prefix cache sharing
        async def sem_generate(p):
            async with semaphore:
                return await self.generate(p, **kwargs)
        return await asyncio.gather(*[sem_generate(p) for p in prompts])

    async def close(self):
        if self._session:
            await self._session.aclose()
            self._session = None
