import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, List, Any

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

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        active_model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        active_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the GRPO clipped surrogate objective and KL penalty.
        """
        # 1. Calculate True Reference Logprobs (Frozen pass)
        with torch.no_grad():
            if hasattr(self.ref_model, "disable_adapter"):
                with self.ref_model.disable_adapter():
                    ref_outputs = self.ref_model(**inputs)
            else:
                ref_outputs = self.ref_model(**inputs)
            ref_logprobs = self._get_logprobs(ref_outputs.logits, inputs["labels"])

        # 2. Probability Ratio
        ratio = torch.exp(active_logprobs - ref_logprobs)
        
        # 3. Clipped Surrogate Objective
        advantages = advantages.unsqueeze(1) # Broadcast across sequence
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # 4. Stable KL Penalty (Masked to prevent NaN from inf * 0)
        log_ratio = ref_logprobs - active_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        kl = torch.where(attention_mask[:, 1:].bool(), kl, torch.zeros_like(kl))
        
        # 5. Combined Loss
        loss = policy_loss + self.kl_coeff * kl
        
        # Mask out prompts and return mean
        loss = loss * attention_mask[:, 1:]
        return loss.sum() / (attention_mask[:, 1:].sum() + 1e-8)

    def update(
        self,
        active_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prompts: List[str],
        rollouts: List[Any],
        rewards: List[float]
    ) -> float:
        """
        Runs one step of PPO/GRPO optimization with mini-batching to prevent OOM.
        """
        active_model.train()
        optimizer.zero_grad()
        
        # Compute global advantages across the entire group
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.ref_model.device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        global_advantages = (rewards_tensor - mean_reward) / std_reward
        
        mini_batch_size = 1
        total_loss = 0.0
        num_mini_batches = max(1, len(prompts) // mini_batch_size)
        
        for i in range(0, len(prompts), mini_batch_size):
            mb_prompts = prompts[i:i+mini_batch_size]
            mb_rollouts = rollouts[i:i+mini_batch_size]
            mb_advantages = global_advantages[i:i+mini_batch_size]
            
            # 1. Tokenize prompts + generations
            full_texts = [p + r.raw_text for p, r in zip(mb_prompts, mb_rollouts)]
            inputs = self.tokenizer(
                full_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1500
            )
            inputs = {k: v.to(self.ref_model.device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            
            # 2. Mask out prompt tokens from loss
            prompt_inputs = self.tokenizer(
                mb_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1500
            )
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
            
            attention_mask = inputs["attention_mask"].clone()
            for j, length in enumerate(prompt_lens):
                attention_mask[j, :length] = 0
                
            # 3. Forward pass to get active policy logprobs
            outputs = active_model(**inputs)
            active_logprobs = self._get_logprobs(outputs.logits, inputs["labels"])
            
            # 4. Compute Loss using pre-computed advantages
            loss = self.compute_loss(
                active_model, 
                inputs, 
                active_logprobs, 
                mb_advantages, 
                attention_mask
            )
            
            # Scale loss for gradient accumulation
            loss = loss / num_mini_batches
            
            # 5. Backpropagate
            loss.backward()
            total_loss += loss.item() * num_mini_batches
            
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.step()
        return total_loss

if __name__ == "__main__":
    print("✅ GRPO Trainer updated for Reference Model Distillation.")
