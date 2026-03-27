"""
Mock environment to verify TurnLevelRewardMixin (MT-GRPO).
"""

import asyncio
from typing import Any, Dict, List
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.turn_level_reward import TurnLevelRewardMixin
from atroposlib.type_definitions import Item

class MockMTGRPOEnv(HermesAgentBaseEnv, TurnLevelRewardMixin):
    async def setup(self):
        # Dummy dataset
        self.dataset = [{"prompt": "Hello", "task_id": "test_1"}]
        self.iter = 0

    async def get_next_item(self) -> Item:
        item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        return item

    def format_prompt(self, item: Item) -> str:
        return item["prompt"]

    async def compute_reward(self, item: Item, result: Any, ctx: Any) -> float:
        # Should not be called because we are using TurnLevelRewardMixin
        return 0.0

    async def compute_turn_rewards(self, item: Item, result: Any, ctx: Any) -> List[float]:
        # Count assistant turns
        assistant_turns = sum(1 for msg in result.messages if msg["role"] == "assistant")
        print(f"DEBUG: Found {assistant_turns} assistant turns. Returning list of 1.0s.")
        return [1.0] * assistant_turns

    async def evaluate(self, *args, **kwargs):
        pass

if __name__ == "__main__":
    MockMTGRPOEnv.cli()
