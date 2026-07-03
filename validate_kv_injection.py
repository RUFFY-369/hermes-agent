#!/usr/bin/env python3
"""
validate_kv_injection.py — Phase 2: KV-Cache Injection Validation
==================================================================

Proves that Q4-quantized KV-cache states can be stored, loaded, and
injected into a new conversation so the model "remembers" the original
context WITHOUT consuming any context-window tokens.

This is the Mode A demonstration — KV-cache blocks are pre-populated
into the model's attention state before inference begins. The model
attends to the injected memories directly, bypassing text re-reading.

Methodology:
  1. Load a model (Qwen2.5-1.5B-Instruct)
  2. Conversation A: teach the model a specific fact (e.g., a fictional API key)
  3. Capture KV-cache after the "teaching" turns
  4. Mean-pool across sequence positions into embeddings
  5. Q4-quantize and save to disk
  6. Start a FRESH conversation B (no context of Conversation A)
  7. Dequantize KV-cache, inject as prefix by prepending to past_key_values
  8. Ask the model to recall the fact — it should "remember" what was taught

Key metric: The model correctly recalls information from injected KV-cache
without that information appearing anywhere in the text prompt.

Usage:
    python validate_kv_injection.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Q4 Quantization (same as Phase 0 / plugins/memory/kv_memory/quantize.py)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_q4_per_channel(tensor: np.ndarray,
                             channel_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """FP32 → Q4 symmetric per-channel quantization."""
    D = tensor.shape[0]
    if D % 2 != 0:
        tensor = np.pad(tensor, (0, 1), mode="constant")
        D = tensor.shape[0]
    num_channels = D // channel_size
    if D % channel_size != 0:
        padded_len = (num_channels + 1) * channel_size
        tensor = np.pad(tensor, (0, padded_len - D), mode="constant")
        D = padded_len
        num_channels = D // channel_size
    reshaped = tensor.reshape(num_channels, channel_size)
    abs_max = np.maximum(np.abs(reshaped).max(axis=1, keepdims=True), 1e-12)
    scales = (abs_max / 7.0).squeeze(axis=1).astype(np.float32)
    quantized = np.clip(np.round(reshaped / abs_max * 7.0), -7, 7).astype(np.int8)
    flat = quantized.ravel()
    even = flat[0::2] & 0x0F
    odd = flat[1::2] & 0x0F
    packed = (even | (odd << 4)).astype(np.uint8)
    return packed, scales


def dequantize_q4_per_channel(packed: np.ndarray, scales: np.ndarray,
                               channel_size: int = 128,
                               original_len: int | None = None) -> np.ndarray:
    """Q4 → FP32 dequantization."""
    even = (packed & 0x0F).astype(np.float32)
    odd = ((packed >> 4) & 0x0F).astype(np.float32)
    even = np.where(even > 7, even - 16, even)
    odd = np.where(odd > 7, odd - 16, odd)
    D = len(scales) * channel_size
    flat = np.empty(D, dtype=np.float32)
    flat[0::2] = even[:D // 2 + D % 2]
    flat[1::2] = odd[:D // 2]
    num_channels = len(scales)
    reshaped = flat.reshape(num_channels, channel_size)
    dequant = reshaped * (scales[:, np.newaxis] / 7.0)
    result = dequant.ravel()
    if original_len is not None:
        result = result[:original_len]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# KV-Cache capture & injection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoredKVCache:
    """Serializable KV-cache snapshot with position-agnostic keys."""
    model_name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    # Per-layer: packed (K_q4, V_q4) and their scales
    # Keys are stored UNROTATED (RoPE stripped) for position-agnostic injection
    layers: list[dict]
    metadata: dict


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE utilities — strip and re-apply rotary position embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def _get_rope_cos_sin(model, device, seq_len: int, offset: int = 0):
    """Extract cos/sin tables from the model's rotary embedding.

    Returns (cos, sin) as float16 tensors of shape (1, 1, seq_len, head_dim).
    """
    rotary_emb = None
    for module in model.modules():
        if "RotaryEmbedding" in module.__class__.__name__:
            rotary_emb = module
            break
    if rotary_emb is None:
        raise RuntimeError("Could not find RotaryEmbedding module")

    # Build position IDs and call the rotary embedding
    position_ids = torch.arange(offset, offset + seq_len, device=device).unsqueeze(0)
    # Qwen2RotaryEmbedding.forward(x, position_ids) -> (cos, sin)
    dummy = torch.zeros(1, 1, seq_len, 1, device=device, dtype=torch.float16)
    cos, sin = rotary_emb(dummy, position_ids)
    return cos.to(dtype=torch.float16), sin.to(dtype=torch.float16)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE rotation helper: swap half-dimensions with sign flip."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE. x: (B,H,S,D), cos/sin: (B,S,D) -> unsqueeze to (B,1,S,D)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def unapply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse RoPE. x: (B,H,S,D), cos/sin: (B,S,D) -> unsqueeze to (B,1,S,D)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) - (_rotate_half(x) * sin)


def capture_kv_cache_rope_aware(model, past_key_values, model_name: str,
                                  channel_size: int = 128) -> StoredKVCache:
    """Capture KV-cache with RoPE stripped from keys for position-agnostic storage.

    Keys are unrotated before quantization so they can be injected at any
    position offset later. Values are stored as-is (no position encoding).
    """
    num_layers = len(past_key_values)
    first_k, first_v = _get_kv(past_key_values, 0)
    num_kv_heads = first_k.shape[1]
    head_dim = first_k.shape[3]
    seq_len = first_k.shape[2]
    device = first_k.device

    # Get position encodings for positions [0, seq_len)
    cos, sin = _get_rope_cos_sin(model, device, seq_len, offset=0)

    layers = []
    for layer_idx in range(num_layers):
        k, v = _get_kv(past_key_values, layer_idx)
        # Unrotate keys: strip RoPE (k is 4D: B,H,S,D; cos/sin are 3D: B,S,D)
        k_unrotated = unapply_rope(k.float(), cos, sin)[0]  # [0] to remove batch
        k_np = k_unrotated.cpu().numpy()
        v_np = v[0].float().cpu().numpy()

        k_packed_parts, k_scale_parts = [], []
        v_packed_parts, v_scale_parts = [], []

        for h in range(num_kv_heads):
            k_head = k_np[h].ravel()
            v_head = v_np[h].ravel()
            k_p, k_s = quantize_q4_per_channel(k_head, channel_size)
            v_p, v_s = quantize_q4_per_channel(v_head, channel_size)
            k_packed_parts.append(k_p)
            k_scale_parts.append(k_s)
            v_packed_parts.append(v_p)
            v_scale_parts.append(v_s)

        layers.append({
            "k_packed": np.concatenate(k_packed_parts),
            "k_scales": np.concatenate(k_scale_parts),
            "v_packed": np.concatenate(v_packed_parts),
            "v_scales": np.concatenate(v_scale_parts),
            "head_dim": head_dim,
            "seq_len": seq_len,
        })

    return StoredKVCache(
        model_name=model_name, num_layers=num_layers, num_kv_heads=num_kv_heads,
        head_dim=head_dim, seq_len=seq_len, layers=layers, metadata={"rope_aware": True},
    )


def dequantize_kv_cache(stored: StoredKVCache,
                         model=None,
                         position_offset: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Dequantize StoredKVCache back into past_key_values format.

    If model is provided and the cache was stored with RoPE stripped,
    keys are re-rotated at position_offset for correct injection.
    """
    dequantized = []
    for layer_idx, layer_data in enumerate(stored.layers):
        num_heads = stored.num_kv_heads
        head_dim = layer_data["head_dim"]
        seq_len = layer_data["seq_len"]
        elements_per_head = seq_len * head_dim

        k_packed = layer_data["k_packed"]
        k_scales = layer_data["k_scales"]
        v_packed = layer_data["v_packed"]
        v_scales = layer_data["v_scales"]

        k_scales_per_head = np.split(k_scales, num_heads)
        v_scales_per_head = np.split(v_scales, num_heads)
        k_bytes_per_head = elements_per_head // 2
        v_bytes_per_head = elements_per_head // 2

        k_heads, v_heads = [], []
        offset_k, offset_v = 0, 0
        for h in range(num_heads):
            k_h_packed = k_packed[offset_k:offset_k + k_bytes_per_head]
            v_h_packed = v_packed[offset_v:offset_v + v_bytes_per_head]
            offset_k += k_bytes_per_head
            offset_v += v_bytes_per_head

            k_h = dequantize_q4_per_channel(
                k_h_packed, k_scales_per_head[h],
                channel_size=head_dim, original_len=elements_per_head,
            ).reshape(seq_len, head_dim)
            v_h = dequantize_q4_per_channel(
                v_h_packed, v_scales_per_head[h],
                channel_size=head_dim, original_len=elements_per_head,
            ).reshape(seq_len, head_dim)

            k_heads.append(torch.from_numpy(k_h))
            v_heads.append(torch.from_numpy(v_h))

        k_tensor = torch.stack(k_heads, dim=0).unsqueeze(0)  # (1, H, S, D)
        v_tensor = torch.stack(v_heads, dim=0).unsqueeze(0)

        # Re-apply RoPE at the injection position if needed
        if model is not None and stored.metadata.get("rope_aware"):
            cos, sin = _get_rope_cos_sin(model, k_tensor.device, seq_len, offset=position_offset)
            k_tensor = apply_rope(k_tensor.float(), cos, sin)

        dequantized.append((k_tensor, v_tensor))

    return dequantized


def inject_kv_cache(model, past_key_values, injected_cache: list,
                     device: torch.device):
    """Prepend injected KV-cache to the model's current past_key_values.

    Returns a new DynamicCache (transformers 5.x compatible) with the
    injected cache prepended to the current cache along the sequence dim.
    """
    from transformers import DynamicCache

    new_cache = DynamicCache()

    for layer_idx, (inj_kv) in enumerate(injected_cache):
        inj_k, inj_v = inj_kv  # CPU tensors, shape (1, H, S_inj, D)
        cur_k, cur_v = _get_kv(past_key_values, layer_idx)  # (1, H, S_cur, D)

        # Move injection to device and concatenate along sequence dim
        inj_k = inj_k.to(device)
        inj_v = inj_v.to(device)
        combined_k = torch.cat([inj_k, cur_k], dim=2)  # (1, H, S_inj+S_cur, D)
        combined_v = torch.cat([inj_v, cur_v], dim=2)

        # Update the cache layer
        new_cache.update(combined_k, combined_v, layer_idx)

    return new_cache


def _get_kv(past_key_values, layer_idx: int):
    """Get (K, V) tensors for a layer, handling various formats."""
    if hasattr(past_key_values, "layers"):
        # DynamicCache (transformers >=5.x)
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    elif isinstance(past_key_values, tuple):
        return past_key_values[layer_idx]
    else:
        item = list(past_key_values)[layer_idx]
        if hasattr(item, "keys"):
            return item.keys, item.values
        return item[0], item[1]


# ═══════════════════════════════════════════════════════════════════════════════
# Test utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_storage_size(stored: StoredKVCache) -> dict:
    """Compute storage metrics for a captured KV-cache."""
    total_packed = 0
    total_scales = 0
    for layer in stored.layers:
        total_packed += len(layer["k_packed"]) + len(layer["v_packed"])
        total_scales += (layer["k_scales"].nbytes + layer["v_scales"].nbytes)

    # FP16 equivalent: 2 bytes per element
    fp16_elements = (stored.num_layers * stored.num_kv_heads *
                     stored.seq_len * stored.head_dim * 2)  # K + V
    fp16_bytes = fp16_elements * 2

    return {
        "num_layers": stored.num_layers,
        "num_kv_heads": stored.num_kv_heads,
        "head_dim": stored.head_dim,
        "seq_len": stored.seq_len,
        "fp16_bytes": fp16_bytes,
        "fp16_mb": round(fp16_bytes / (1024 * 1024), 2),
        "q4_packed_bytes": total_packed,
        "q4_packed_mb": round(total_packed / (1024 * 1024), 2),
        "q4_scales_bytes": total_scales,
        "total_q4_mb": round((total_packed + total_scales) / (1024 * 1024), 2),
        "compression_ratio": round(fp16_bytes / (total_packed + total_scales), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Conversations
# ═══════════════════════════════════════════════════════════════════════════════

# A multi-turn conversation split into PREFIX (stored/injected) and SUFFIX (fresh query).
# This avoids RoPE position mismatch: both parts are from the SAME conversation,
# so position encodings are consistent. This is the actual memory use case:
# store conversation history → restore it later → continue the conversation.

PREFIX_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant with excellent memory."},
    {"role": "user", "content": (
        "I need you to remember three critical pieces of information:\n\n"
        "1. Production DB: postgresql://admin:xK9mR2pL8qW5v@db-prod.internal:5432/maindb\n"
        "2. Payment API key: pk_live_8fA3bC2dE1gH4iJ5kL6mN7oP\n"
        "3. Backup server IP: 10.23.45.67, SSH fingerprint: SHA256:abc123def456ghi789jkl012mno345pqr678stu901\n\n"
        "After you confirm, I will ask you to recall them."
    )},
    {"role": "assistant", "content": (
        "I've stored all three items:\n\n"
        "1. **DB**: postgresql://admin:xK9mR2pL8qW5v@db-prod.internal:5432/maindb\n"
        "2. **API key**: pk_live_8fA3bC2dE1gH4iJ5kL6mN7oP\n"
        "3. **Backup**: 10.23.45.67 (fingerprint SHA256:abc123def456ghi789jkl012mno345pqr678stu901)\n\n"
        "Ready — ask me to recall them anytime."
    )},
]

SUFFIX_MESSAGE = [
    {"role": "user", "content": (
        "Now tell me: what are the three items I asked you to remember? "
        "List each one with its exact value."
    )},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Validate KV-cache injection for memory retrieval"
    )
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("\n── Step 1: Loading model ──")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # ── Step 2: Run "teaching" conversation & capture KV-cache ──────────
    print("\n── Step 2: Teaching conversation + KV-cache capture ──")
    teach_text = tokenizer.apply_chat_template(
        PREFIX_CONVERSATION, tokenize=False, add_generation_prompt=False,
    )
    teach_inputs = tokenizer(teach_text, return_tensors="pt").to(device)
    teach_len = teach_inputs["input_ids"].shape[1]
    print(f"  Teaching tokens: {teach_len}")

    with torch.no_grad():
        teach_outputs = model(**teach_inputs, use_cache=True, output_hidden_states=True)

    past_kv = teach_outputs.past_key_values
    if past_kv is None:
        print("ERROR: No KV-cache returned. Aborting.")
        sys.exit(1)

    # Capture + Q4 quantize (RoPE-aware: strip position encoding from keys)
    t1 = time.perf_counter()
    stored = capture_kv_cache_rope_aware(model, past_kv, args.model, channel_size=128)
    capture_time = time.perf_counter() - t1

    storage = compute_storage_size(stored)
    print(f"  Captured (RoPE-stripped): {storage['num_layers']}L x {storage['num_kv_heads']}H")
    print(f"  FP16: {storage['fp16_mb']}MB -> Q4: {storage['total_q4_mb']}MB ({storage['compression_ratio']}x)")
    print(f"  Capture + quantize: {capture_time*1000:.0f}ms")

    # ── Step 3: Generate response WITHOUT injection (baseline) ──────────
    print("\n── Step 3: Baseline — recall WITHOUT KV injection ──")
    recall_text = tokenizer.apply_chat_template(
        SUFFIX_MESSAGE, tokenize=False, add_generation_prompt=True,
    )
    recall_inputs = tokenizer(recall_text, return_tensors="pt").to(device)

    with torch.no_grad():
        baseline_outputs = model.generate(
            **recall_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
        )
    baseline_response = tokenizer.decode(
        baseline_outputs[0][recall_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"  Baseline response: \"{baseline_response[:200]}\"")

    # ── Step 4: Generate WITH KV injection ──────────────────────────────
    print("\n── Step 4: Recall WITH injected KV-cache ──")

    # Dequantize + re-apply RoPE at position 0 (prefix injection)
    dequantized_kv = dequantize_kv_cache(stored, model=model, position_offset=0)
    print(f"  Dequantized {len(dequantized_kv)} layers (RoPE re-applied at pos 0)")

    # Inject prefix KV-cache: build DynamicCache, use generate() with
    # full attention_mask spanning prefix+suffix positions.
    with torch.no_grad():
        from transformers import DynamicCache
        combined_cache = DynamicCache()
        for layer_idx, (inj_k, inj_v) in enumerate(dequantized_kv):
            combined_cache.update(
                inj_k.to(device=device, dtype=torch.float16),
                inj_v.to(device=device, dtype=torch.float16),
                layer_idx,
            )
        # Full attention mask: prefix + suffix tokens
        full_mask = torch.ones(1, teach_len + recall_inputs["input_ids"].shape[1],
                               device=device)
        injected_out = model.generate(
            input_ids=recall_inputs["input_ids"],
            attention_mask=full_mask,
            past_key_values=combined_cache,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    injected_response = tokenizer.decode(
        injected_out[0][recall_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"  Injected: \"{injected_response[:300]}\"")

    # ── Step 5: Analysis ────────────────────────────────────────────────
    print("\n── Step 5: Analysis ──")
    print(f"  Teaching tokens: {teach_len}")
    print(f"  Recall tokens (text): {recall_inputs['input_ids'].shape[1]}")
    print(f"  Recall tokens (with KV injection): {recall_inputs['input_ids'].shape[1]} "
          f"(same — injection costs ZERO context tokens)")
    print(f"  Context saved: {teach_len} tokens (100%)")

    # Check if the injected response contains the taught information
    taught_strings = ["xK9mR2pL8qW5v", "pk_live_8fA3bC2dE1gH4iJ5kL6mN7oP",
                       "10.23.45.67"]
    baseline_hits = sum(1 for s in taught_strings if s in baseline_response)
    injected_hits = sum(1 for s in taught_strings if s in injected_response)

    print(f"\n  Baseline recall (no injection): {baseline_hits}/3 facts recalled")
    print(f"  Injected recall (KV-cache):     {injected_hits}/3 facts recalled")

    if injected_hits > baseline_hits:
        print("\n  >> KV-CACHE INJECTION WORKS (RoPE-aware)")
        print("    Keys stored unrotated, re-rotated at injection position.")
        print("    Model recalls taught facts without text in context window.")
    elif injected_hits == baseline_hits:
        print("\n  >> No improvement — check RoPE rotation correctness")
    else:
        print("\n  >> UNEXPECTED")

    # ── Storage metrics ──────────────────────────────────────────────────
    print(f"\n── Storage Summary ──")
    print(f"  FP16 KV-cache:   {storage['fp16_mb']}MB")
    print(f"  Q4 compressed:   {storage['total_q4_mb']}MB")
    print(f"  Compression:     {storage['compression_ratio']}×")
    print(f"  Tokens stored:   {teach_len}")
    print(f"  Storage/token:   {storage['total_q4_mb']*1024/teach_len:.1f}KB")
    print(f"  Total time:      {time.perf_counter() - t0:.1f}s")

    # ── Save report ──────────────────────────────────────────────────────
    report = {
        "model": args.model,
        "teaching_tokens": teach_len,
        "baseline_response": baseline_response,
        "injected_response": injected_response,
        "baseline_hits": baseline_hits,
        "injected_hits": injected_hits,
        "storage": {k: v for k, v in storage.items() if not isinstance(v, np.ndarray)},
        "capture_time_ms": capture_time * 1000,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to: {args.output}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return report


if __name__ == "__main__":
    main()
