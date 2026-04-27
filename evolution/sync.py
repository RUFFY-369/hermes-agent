import asyncio
import aiohttp
import os
import json
from typing import Optional

class LoRASyncEngine:
    """
    Parameter Server utility for SGLang.
    Hot-swaps LoRA adapters in the inference pool without dropping KV-cache.
    """
    def __init__(self, sgl_base_url: str = "http://localhost:30000"):
        self.base_url = sgl_base_url
        self.load_endpoint = f"{sgl_base_url}/load_lora_adapter"
        self.unload_endpoint = f"{sgl_base_url}/unload_lora_adapter"

    async def sync_weights(self, adapter_path: str, adapter_name: str = "hermes-rl-active"):
        """
        Hot-swaps the active LoRA adapter in SGLang.
        
        Args:
            adapter_path: Absolute path to the LoRA adapter directory (must be accessible by SGLang server).
            adapter_name: Unique identifier for the adapter in SGLang's pool.
        """
        payload = {
            "lora_path": adapter_path,
            "lora_name": adapter_name
        }

        async with aiohttp.ClientSession() as session:
            # Attempt to unload if exists to ensure a clean swap
            try:
                await session.post(self.unload_endpoint, json={"lora_name": adapter_name})
            except Exception:
                pass

            # Load the new weights
            async with session.post(self.load_endpoint, json=payload) as resp:
                if resp.status == 200:
                    print(f"✅ LoRA Sync Success: Adapter '{adapter_name}' updated from {adapter_path}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"❌ LoRA Sync Failed ({resp.status}): {error_text}")
                    return False

if __name__ == "__main__":
    async def verify():
        engine = LoRASyncEngine()
        print("🚀 Testing LoRA Sync Bridge...")
        # Note: This will fail if SGLang is not running or doesn't have the path
        await engine.sync_weights("/home/ruffy-369/NousResearch/hermes-rl/output/adapter_v1", "test_adapter")

    asyncio.run(verify())
