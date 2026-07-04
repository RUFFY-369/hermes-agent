"""Auxiliary LLM client for Evolution Engine analysis and proposals.

Uses a separate, cheaper model for evolution work to avoid polluting the
main conversation's prompt cache. Follows the existing auxiliary_client.py
pattern but is specialized for evolution tasks.

Configuration (config.yaml):
  auxiliary:
    evolution:
      provider: deepseek
      model: deepseek-chat
      api_key: sk-xxx  # or read from DEEPSEEK_API_KEY env var
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default evolution model — cheap and fast for analysis work
DEFAULT_EVOLUTION_PROVIDER = "deepseek"
DEFAULT_EVOLUTION_MODEL = "deepseek-chat"
DEFAULT_EVOLUTION_BASE_URL = "https://api.deepseek.com"

# Token limits
MAX_OUTPUT_TOKENS = 4096
MAX_RETRIES = 2


class EvolutionLLMClient:
    """Lightweight LLM client for evolution analysis and proposal generation.

    Uses the OpenAI-compatible API so it works with DeepSeek, OpenRouter,
    or any compatible provider.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_EVOLUTION_BASE_URL)
        self.model = model or os.getenv("EVOLUTION_MODEL", DEFAULT_EVOLUTION_MODEL)
        self._client = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                logger.warning("openai package not available for evolution LLM")
                return None
        return self._client

    async def analyze(self, prompt: str) -> str:
        """Run failure analysis. Returns JSON string response."""
        return await self._call(prompt, "failure analysis")

    async def propose(self, prompt: str) -> str:
        """Generate improvement proposals. Returns JSON string response."""
        return await self._call(prompt, "improvement proposal")

    async def judge(self, rubric: str, trajectory_json: str) -> Dict[str, Any]:
        """LLM-as-judge for qualitative evaluation criteria."""
        prompt = f"""You are an evaluation judge for an AI agent. Score the agent's performance against the rubric below.

RUBRIC:
{rubric}

AGENT TRAJECTORY:
{trajectory_json[:8000]}

Respond with ONLY valid JSON:
{{"passed": true/false, "score": 0.0-1.0, "reasoning": "<brief explanation>"}}"""

        response = await self._call(prompt, "llm judge")
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"passed": False, "score": 0.0, "reasoning": f"Failed to parse judge response: {response[:200]}"}

    async def _call(self, prompt: str, task_label: str = "") -> str:
        """Make an LLM API call with retries."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("Evolution LLM client not available (no API key)")

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an AI engineering assistant. Respond with ONLY valid JSON. No markdown, no explanations outside the JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.3,  # Low temperature for analytical work
                )
                content = response.choices[0].message.content
                return content or "{}"
            except Exception as e:
                last_error = e
                logger.warning(
                    "Evolution LLM call failed (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES + 1, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1.0 * (attempt + 1))  # Linear backoff

        raise RuntimeError(f"Evolution LLM call failed after {MAX_RETRIES + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


_evolution_llm: Optional[EvolutionLLMClient] = None


def get_evolution_llm() -> Optional[EvolutionLLMClient]:
    """Return the process-wide evolution LLM client singleton."""
    global _evolution_llm
    if _evolution_llm is None:
        # Try to resolve from config
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            try:
                from hermes_cli.config import load_config
                cfg = load_config()
                aux = cfg.get("auxiliary", {}) if isinstance(cfg, dict) else {}
                evo = aux.get("evolution", {}) if isinstance(aux, dict) else {}
                api_key = evo.get("api_key", "") or os.getenv("DEEPSEEK_API_KEY", "")
            except Exception:
                pass
        if api_key:
            _evolution_llm = EvolutionLLMClient(api_key=api_key)
    return _evolution_llm
