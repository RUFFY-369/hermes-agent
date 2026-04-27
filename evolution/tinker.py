import json
import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Internal import from hermes-agent root
try:
    from tools.rl_training_tool import (
        rl_select_environment,
        rl_edit_config,
        rl_start_training,
        rl_check_status,
        rl_get_current_config
    )
except ImportError:
    print("⚠️  Warning: Could not import hermes-agent RL tools.")
    rl_start_training = None

logger = logging.getLogger(__name__)

class TinkerBridgeTrainer:
    """
    Hybrid Trainer Bridge.
    Offloads the heavy weight-update (backward pass) to the Tinker API 
    while keeping rollouts local in SGLang.
    """
    def __init__(self, use_tinker: bool = True):
        self.use_tinker = use_tinker and (rl_start_training is not None)
        self.current_run_id = None
        self.tinker_api_key = os.getenv("TINKER_API_KEY")
        
        if self.use_tinker and not self.tinker_api_key:
            print("⚠️  TINKER_API_KEY missing. Falling back to local training.")
            self.use_tinker = False

    async def train_step(self, rewards: List[float], rollouts: List[Any], task: str) -> Optional[str]:
        """
        Executes a training step via Tinker.
        """
        if not self.use_tinker:
            return None

        print(f"☁️  [TinkerBridge] Offloading training step to Tinker Cloud...")
        
        await rl_select_environment("coding_gen") 
        await rl_edit_config("total_steps", 100) 
        await rl_edit_config("learning_rate", 2e-5)
        
        res_json = await rl_start_training()
        res = json.loads(res_json)
        
        if "error" in res:
            return None
            
        self.current_run_id = res["run_id"]
        
        while True:
            status_json = await rl_check_status(self.current_run_id)
            status = json.loads(status_json)
            curr_status = status.get("status")
            
            if curr_status in ["completed", "stopped"]:
                return f"/workspace/hermes-rl/output/tinker_{self.current_run_id}"
            elif curr_status == "failed":
                return None
                
            await asyncio.sleep(60) 

    def is_active(self) -> bool:
        return self.use_tinker
