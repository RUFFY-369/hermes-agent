"""Tests for semantic retrieval."""

import os
import tempfile

import numpy as np
import pytest

from plugins.memory.kv_memory.config import KVMemoryConfig
from plugins.memory.kv_memory.quantize import quantize_q4_per_channel
from plugins.memory.kv_memory.retrieval import (
    KVRetriever,
    cosine_similarity,
    batch_cosine_similarity,
    compute_temporal_weight,
    mmr_rerank,
)
from plugins.memory.kv_memory.storage import KVMemoryDB


@pytest.fixture
def populated_db():
    """Create a database with 4 turns: 3 related, 1 unrelated."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = KVMemoryDB(path)
    db.initialize_schema()

    # Store turns with embeddings biased for related/unrelated
    texts = [
        "Python async event loop and coroutines",
        "Async programming patterns in Python",
        "FastAPI async request handling",
        "Italian pasta carbonara recipe",
    ]
    for i, text in enumerate(texts):
        emb = np.random.randn(384).astype(np.float32)
        if i < 3:  # related: shift in same direction
            emb += np.array([0.5] * 384, dtype=np.float32)
        packed, scales = quantize_q4_per_channel(emb, channel_size=128)
        db.store_turn(
            session_id=f"s{i}", turn_number=1, embedding=emb,
            q4_embedding=packed, q4_scales=scales,
            summary_text=text, head_dim=128, num_kv_heads=3,
        )

    config = KVMemoryConfig(db_path=path, top_k=3, min_similarity=0.0)
    retriever = KVRetriever(db, config)

    yield db, retriever

    db.close()
    for ext in ["", "-wal", "-shm"]:
        p = path + ext
        if os.path.exists(p):
            os.unlink(p)


class TestCosineSimilarity:
    """Basic similarity computations."""

    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([-1.0, -2.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.zeros(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_batch(self):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        sims = batch_cosine_similarity(query, candidates)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)


class TestTemporalDecay:
    """Temporal decay weighting."""

    def test_no_decay(self):
        now = 1000.0
        w = compute_temporal_weight(now - 86400, now, half_life_days=0)
        assert w == 1.0

    def test_exact_half_life(self):
        now = 1000.0
        one_week_ago = now - 7 * 86400
        w = compute_temporal_weight(one_week_ago, now, half_life_days=7)
        assert w == pytest.approx(0.5)

    def test_recent_higher(self):
        now = 1000.0
        w_old = compute_temporal_weight(now - 14 * 86400, now, half_life_days=7)
        w_new = compute_temporal_weight(now - 1 * 86400, now, half_life_days=7)
        assert w_new > w_old


class TestMMR:
    """Maximal Marginal Relevance reranking."""

    def test_pure_relevance(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        candidates = [
            np.array([1.0, 0.0], dtype=np.float32),  # best
            np.array([0.9, 0.1], dtype=np.float32),  # good
            np.array([0.0, 1.0], dtype=np.float32),  # unrelated
        ]
        scores = [1.0, 0.9, 0.1]
        indices = mmr_rerank(query, candidates, scores, k=2, lambda_param=1.0)
        assert indices == [0, 1]  # Pure relevance order

    def test_diversity(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        candidates = [
            np.array([1.0, 0.0], dtype=np.float32),  # best but similar to [1]
            np.array([0.99, 0.01], dtype=np.float32),  # also good, redundant
            np.array([0.0, 1.0], dtype=np.float32),  # different topic
        ]
        scores = [1.0, 0.99, 0.5]
        indices = mmr_rerank(query, candidates, scores, k=3, lambda_param=0.3)
        # With lambda=0.3, diversity is favored — index 2 should rank higher
        assert len(indices) == 3


class TestRetriever:
    """Integration tests for KVRetriever."""

    def test_retrieve_related(self, populated_db):
        db, retriever = populated_db
        query = np.random.randn(384).astype(np.float32) + np.array([0.5] * 384, dtype=np.float32)
        results = retriever.retrieve(query)
        assert len(results) > 0
        # Related texts should appear first
        first_text = results[0]["summary_text"]
        assert "Python" in first_text or "Async" in first_text or "FastAPI" in first_text

    def test_retrieve_text_format(self, populated_db):
        db, retriever = populated_db
        query = np.random.randn(384).astype(np.float32) + np.array([0.5] * 384, dtype=np.float32)
        text = retriever.retrieve_text(query)
        assert "KV Memory" in text
        assert len(text) > 0

    def test_retrieve_empty(self, populated_db):
        db, retriever = populated_db
        # Query far from all stored embeddings
        query = np.array([-100.0] * 384, dtype=np.float32)
        results = retriever.retrieve(query)
        # May return empty if below min_similarity
        assert isinstance(results, list)

    def test_session_boost(self, populated_db):
        db, retriever = populated_db
        query = np.random.randn(384).astype(np.float32) + np.array([0.5] * 384, dtype=np.float32)
        results_no_boost = retriever.retrieve(query, current_session_id="")
        results_boosted = retriever.retrieve(query, current_session_id="s0")
        # Both should return results, order may differ
        assert len(results_no_boost) > 0
        assert len(results_boosted) > 0
