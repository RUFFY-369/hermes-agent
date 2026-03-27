"""
TurnLevelRewardMixin -- Multi-Turn Reward Support for MT-GRPO

By default, Atropos environments provide a single scalar reward at the end 
of a rollout. This mixin allows HermesAgentBaseEnv subclasses to return 
a list of rewards (one per assistant turn), which is required for 
Multi-Turn Group Relative Policy Optimization (MT-GRPO).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TurnLevelRewardMixin(ABC):
    """
    Mixin for environments that support turn-level rewards.

    When a HermesAgentBaseEnv subclass inherits from this mixin,
    the base environment will call `compute_turn_rewards()` instead
    of the standard `compute_reward()`.
    """

    @abstractmethod
    async def compute_turn_rewards(
        self, item: Any, result: Any, ctx: Any
    ) -> List[float]:
        """
        Compute a list of rewards, one for each assistant turn.

        Args:
            item: The original dataset item
            result: The AgentResult from the agent loop
            ctx: The ToolContext for verification

        Returns:
            List[float] of rewards, length must match the number of assistant turns.
        """
        raise NotImplementedError
