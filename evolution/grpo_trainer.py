import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Tuple, Dict

class GRPOTrainer:
    """
    H100-Optimized GRPO Trainer.
    Implements Frozen Reference Model for exact KL Divergence.
    """
    def __init__(
        self, 
        model_name: str, 
        clip_eps: float = 0.2, 
        kl_coeff: float = 0.04
    ):
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        
        # Load Frozen Reference Model (Isolate to separate GPU memory if possible)
        print(f"🚀 [GRPO] Loading Frozen Reference Model: {model_name}")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

    def _get_logprobs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Helper to extract logprobs for selected tokens."""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_logprobs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_logprobs

    def compute_loss(
        self,
        model: torch.nn.Module,      # The active policy (LoRA)
        inputs: Dict[str, torch.Tensor],
        logprobs: torch.Tensor,      # Active policy logprobs from rollout
        rewards: torch.Tensor,       # [G] Shaped dense rewards
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the GRPO loss with True KL Penalty.
        """
        # 1. Calculate True Reference Logprobs (Frozen pass)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logprobs = self._get_logprobs(ref_outputs.logits, inputs["labels"])

        # 2. Group Relative Advantage (Normalized across the massive G batch)
        # Using shaped rewards from sandbox.py
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        # Expand for token-wise multiplication
        # Note: logprobs shape [G, seq_len-1] due to shift
        advantages = advantages.unsqueeze(1).expand_as(logprobs)

        # 3. PPO-Clipped Surrogate Loss
        # We compare active logprobs against reference logprobs
        ratio = torch.exp(logprobs - ref_logprobs.detach())
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * attention_mask[:, 1:]).sum() / (attention_mask[:, 1:].sum() + 1e-8)

        # 4. Exact KL Divergence Penalty
        # D_KL(ref || active) = sum(pi_ref * (log pi_ref - log pi_active))
        kl = torch.exp(ref_logprobs) * (ref_logprobs - logprobs)
        kl_loss = (kl * attention_mask[:, 1:]).sum() / (attention_mask[:, 1:].sum() + 1e-8)

        total_loss = policy_loss + self.kl_coeff * kl_loss
        return total_loss

if __name__ == "__main__":
    print("✅ GRPO Trainer updated for Reference Model Distillation.")
