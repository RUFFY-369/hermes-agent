import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any
from serving.sync import LoRASyncEngine

class OPDTrainer:
    """
    On-Policy Distillation (OPD) Trainer.
    Aligns Student to Teacher via token-level KL advantage.
    Also handles the 'Hot-Swap' synchronization with the SGLang server.
    """
    def __init__(self, lr: float = 1e-5):
        self.lr = lr
        self.sync_engine = LoRASyncEngine(sgl_base_url="http://localhost:30000")

    def compute_opd_loss(
        self,
        teacher_logprobs: torch.Tensor, # log pi_teacher(a|s + hint) [B, L]
        student_logprobs: torch.Tensor, # log pi_student(a|s) [B, L]
        attention_mask: torch.Tensor    # [B, L]
    ) -> torch.Tensor:
        """
        Calculates the OPD Distillation Loss.
        Math: Loss = -mean((log_pi_teacher - log_pi_student) * log_pi_student)
        """
        # 1. Dimensionality Assertions
        batch_size, seq_len = teacher_logprobs.shape
        assert student_logprobs.shape == (batch_size, seq_len), "Shape mismatch"
        assert attention_mask.shape == (batch_size, seq_len), "Mask mismatch"

        # 2. Token-level Advantage Calculation
        # A_t = log_pi_teacher - log_pi_student
        with torch.no_grad():
            advantages = teacher_logprobs - student_logprobs  # Shape: [B, L]
        
        # 3. Weighted Distillation Loss
        # Minimize - (A_t * log_pi_student)
        weighted_logprobs = advantages * student_logprobs
        
        # Apply mask and compute mean
        masked_loss = -(weighted_logprobs * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        
        return masked_loss

    async def sync(self, adapter_path: str = "/workspace/hermes-rl/output/adapter_active"):
        """
        Synchronizes the SGLang inference engine with the newly saved PyTorch weights.
        """
        print("🔄 [OPD] Triggering LoRA Hot-Swap...")
        # Use 'active_policy' as the persistent name in SGLang
        success = await self.sync_engine.sync_weights(
            adapter_path=adapter_path, 
            adapter_name="active_policy"
        )
        if success:
            print("✅ [OPD] SGLang Inference Engine Synchronized.")
        else:
            print("⚠️ [OPD] SGLang Sync Failed. Iteration 2 may use stale weights.")
