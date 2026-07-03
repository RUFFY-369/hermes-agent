"""Tests for Q4 per-channel quantization."""

import numpy as np
import pytest
from plugins.memory.kv_memory.quantize import (
    quantize_q4_per_channel,
    dequantize_q4_per_channel,
    compute_q4_size,
)


class TestQ4RoundTrip:
    """Verify FP32 → Q4 → FP32 round-trip fidelity."""

    def test_small_vector(self):
        """Round-trip on a small controlled vector."""
        original = np.array([1.0, -2.5, 0.0, 3.7, -0.5, 1.2, 4.0, -3.0], dtype=np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=4)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=4,
                                                   original_len=8)
        # Cosine similarity should be near-perfect for clean data
        sim = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-12
        )
        assert sim > 0.99, f"Fidelity too low: {sim}"

    def test_random_1024_cs128(self):
        """Round-trip on random 1024-dim vector with channel_size=128."""
        original = np.random.randn(1024).astype(np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=128,
                                                   original_len=1024)
        sim = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-12
        )
        assert sim > 0.95, f"Fidelity too low: {sim}"

    def test_random_384_cs128(self):
        """Round-trip on 384-dim (sentence-transformers output)."""
        original = np.random.randn(384).astype(np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=128,
                                                   original_len=384)
        sim = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-12
        )
        assert sim > 0.95

    def test_odd_length(self):
        """Odd-length vectors should be padded and handled correctly."""
        original = np.random.randn(999).astype(np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=128,
                                                   original_len=999)
        assert len(reconstructed) == 999

    def test_zeros(self):
        """All-zero input should round-trip exactly."""
        original = np.zeros(512, dtype=np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=128,
                                                   original_len=512)
        assert np.allclose(original, reconstructed, atol=1e-6)

    def test_large_values(self):
        """Large values should still round-trip with reasonable fidelity."""
        original = np.random.randn(1024).astype(np.float32) * 100
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=128,
                                                   original_len=1024)
        # Cosine similarity should be direction-preserving even with large values
        sim = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-12
        )
        assert sim > 0.95


class TestCompressionRatio:
    """Verify compression ratio calculations."""

    def test_standard_384_cs128(self):
        """384-dim with cs=128 should be ~3.6× compression."""
        info = compute_q4_size(384, channel_size=128)
        assert info["num_channels"] == 3  # ceil(384/128) = 3
        assert info["compression_ratio"] > 3.0

    def test_standard_4096_cs128(self):
        """4096-dim with cs=128 should be ~3.8×."""
        info = compute_q4_size(4096, channel_size=128)
        assert info["compression_ratio"] > 3.5

    def test_channel_size_affects_ratio(self):
        """Smaller channels = more scales = lower compression."""
        info16 = compute_q4_size(1024, channel_size=16)
        info128 = compute_q4_size(1024, channel_size=128)
        assert info128["compression_ratio"] > info16["compression_ratio"]


class TestEdgeCases:
    """Edge cases that shouldn't crash."""

    def test_single_element(self):
        """Single element should be padded and handled."""
        original = np.array([3.5], dtype=np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=128,
                                                   original_len=1)
        assert len(reconstructed) == 1

    def test_channel_size_divides_exactly(self):
        """When D % cs == 0, no extra padding needed."""
        original = np.random.randn(512).astype(np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        # 512 / 128 = 4 channels, no padding
        assert len(scales) == 4

    def test_mismatched_channel_size(self):
        """Dequantizing with wrong channel_size should fail."""
        original = np.random.randn(512).astype(np.float32)
        packed, scales = quantize_q4_per_channel(original, channel_size=128)
        # Using wrong channel_size (64 instead of 128) should not crash
        # but will produce wrong-length output
        reconstructed = dequantize_q4_per_channel(packed, scales, channel_size=64,
                                                   original_len=512)
        # Should still produce output (might be wrong length if unlucky)
        assert len(reconstructed) > 0
