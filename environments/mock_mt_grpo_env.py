"""
Mock environment to verify TurnLevelRewardMixin (MT-GRPO).
"""

import asyncio
from typing import Any, Dict, List
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.turn_level_reward import TurnLevelRewardMixin
from atroposlib.type_definitions import Item
from atroposlib.envs.server_handling.server_manager import APIServerConfig

class MockMTGRPOEnv(HermesAgentBaseEnv, TurnLevelRewardMixin):
    @classmethod
    def config_init(cls):
        """Override config initialization to set defaults for testing."""
        env_config, _ = super().config_init()
        
        # Set defaults to avoid long CLI strings
        env_config.max_agent_turns = 2
        env_config.group_size = 1
        env_config.data_path_to_save_groups = "mt_grpo_test.jsonl"
        
        # Point to the active vLLM server and enable stabilization
        # Need to set extra_body here so HermesAgentBaseEnv passes it to the agent loop
        env_config.extra_body = {"atropos_inhibit_tools": True}
        
        server_configs = [
            APIServerConfig(
                base_url="http://localhost:9001/v1",
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                server_type="openai",
                extra_body={"atropos_inhibit_tools": True}
            )
        ]
        
        return env_config, server_configs

    async def setup(self):
        # Dummy dataset for sampling
        self.dataset = [{"prompt": "Tell me a joke.", "task_id": "test_1"}]
        self.iter = 0

    async def get_next_item(self) -> Item:
        item = self.dataset[0]
        return item

    def format_prompt(self, item: Item) -> str:
        return item["prompt"]

    async def compute_reward(self, item: Item, result: Any, ctx: Any) -> float:
        # Fallback for non-mixin calls
        return 0.5

    async def compute_turn_rewards(self, item: Item, result: Any, ctx: Any) -> List[float]:
        # Count assistant turns and return one reward per turn
        assistant_turns = sum(1 for msg in result.messages if msg["role"] == "assistant")
        print(f"\n[DEBUG] PR 2 Verification: Found {assistant_turns} assistant turns.")
        rewards = [float(i + 1) for i in range(assistant_turns)]
        print(f"[DEBUG] Returning turn rewards: {rewards}")
        return rewards

    async def evaluate(self, *args, **kwargs):
        pass

if __name__ == "__main__":
    MockMTGRPOEnv.cli()
