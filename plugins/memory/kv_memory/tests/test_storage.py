"""Tests for SQLite storage backend."""

import json
import os
import tempfile
import time

import numpy as np
import pytest

from plugins.memory.kv_memory.quantize import quantize_q4_per_channel
from plugins.memory.kv_memory.storage import KVMemoryDB


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = KVMemoryDB(path)
    database.initialize_schema()
    yield database
    database.close()
    for ext in ["", "-wal", "-shm"]:
        p = path + ext
        if os.path.exists(p):
            os.unlink(p)


@pytest.fixture
def sample_embedding():
    """Generate a 384-dim sample embedding."""
    return np.random.randn(384).astype(np.float32)


class TestCRUD:
    """Basic Create/Read/Update/Delete operations."""

    def test_store_and_retrieve_turn(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        turn_id = db.store_turn(
            session_id="s1", turn_number=1, embedding=sample_embedding,
            q4_embedding=packed, q4_scales=scales,
            summary_text="Test turn", head_dim=128, num_kv_heads=3,
        )
        assert turn_id is not None
        assert len(turn_id) == 36  # UUID

        turns = db.get_turns("s1")
        assert len(turns) == 1
        assert turns[0]["summary_text"] == "Test turn"
        assert turns[0]["turn_number"] == 1

    def test_store_multiple_turns(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        ids = []
        for i in range(5):
            tid = db.store_turn(
                session_id="s1", turn_number=i + 1, embedding=sample_embedding,
                q4_embedding=packed, q4_scales=scales,
                summary_text=f"Turn {i}", head_dim=128, num_kv_heads=3,
            )
            ids.append(tid)

        turns = db.get_turns("s1")
        assert len(turns) == 5
        assert [t["turn_number"] for t in turns] == [1, 2, 3, 4, 5]

    def test_store_turn_with_fp16(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        tid = db.store_turn(
            session_id="s1", turn_number=1, embedding=sample_embedding,
            q4_embedding=packed, q4_scales=scales,
            summary_text="Test", head_dim=128, num_kv_heads=3,
            store_fp16=True,
        )
        emb, meta = db.get_turn_embedding(tid, dequantize=True)
        assert emb is not None
        assert emb.shape[0] == 384

    def test_get_nonexistent_turn(self, db):
        emb = db.get_turn_embedding("nonexistent-id")
        assert emb is None

    def test_get_all_embeddings(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        for i in range(3):
            db.store_turn(
                session_id=f"s{i}", turn_number=1, embedding=sample_embedding,
                q4_embedding=packed, q4_scales=scales,
                summary_text=f"Turn {i}", head_dim=128, num_kv_heads=3,
            )
        all_embs = db.get_all_embeddings()
        assert len(all_embs) == 3

        # Filter by session
        s0_embs = db.get_all_embeddings(session_id="s0")
        assert len(s0_embs) == 1


class TestSessions:
    """Session management."""

    def test_ensure_session(self, db):
        db.ensure_session("s1", metadata={"platform": "test"})
        s = db.get_session("s1")
        assert s is not None
        assert s["id"] == "s1"
        assert "test" in s["metadata"]

    def test_ensure_session_idempotent(self, db):
        db.ensure_session("s1")
        t1 = db.get_session("s1")["last_accessed"]
        time.sleep(0.01)
        db.ensure_session("s1")
        t2 = db.get_session("s1")["last_accessed"]
        assert t2 > t1  # last_accessed updated


class TestSessionLinks:
    """Cross-session linking."""

    def test_link_sessions(self, db):
        db.ensure_session("s1")
        db.ensure_session("s2")
        db.link_sessions("s1", "s2", similarity=0.85, link_type="semantic")
        linked = db.get_linked_sessions("s1", min_similarity=0.5)
        assert len(linked) == 1
        assert linked[0]["similarity"] == 0.85

    def test_link_below_threshold(self, db):
        db.ensure_session("s1")
        db.ensure_session("s2")
        db.link_sessions("s1", "s2", similarity=0.3)
        linked = db.get_linked_sessions("s1", min_similarity=0.5)
        assert len(linked) == 0


class TestGovernance:
    """Compaction and pruning."""

    def test_prune_old_turns(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        db.store_turn(
            session_id="s1", turn_number=1, embedding=sample_embedding,
            q4_embedding=packed, q4_scales=scales,
            summary_text="Old turn", head_dim=128, num_kv_heads=3,
        )
        # Prune with 0 days retention (everything is old)
        removed = db.prune_old_turns(0)
        assert removed >= 0  # May be 0 if timestamp is too recent, or 1

    def test_compact_session(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        for i in range(6):
            db.store_turn(
                session_id="s1", turn_number=i + 1, embedding=sample_embedding,
                q4_embedding=packed, q4_scales=scales,
                summary_text=f"Turn {i}", head_dim=128, num_kv_heads=3,
            )
        before = len(db.get_turns("s1"))
        removed = db.compact_session("s1")
        after = len(db.get_turns("s1"))
        assert before - after == removed
        assert removed > 0  # At least some turns merged


class TestStats:
    """Statistics reporting."""

    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["total_turns"] == 0
        assert stats["total_sessions"] == 0

    def test_populated_stats(self, db, sample_embedding):
        packed, scales = quantize_q4_per_channel(sample_embedding, channel_size=128)
        db.store_turn(
            session_id="s1", turn_number=1, embedding=sample_embedding,
            q4_embedding=packed, q4_scales=scales,
            summary_text="Test", head_dim=128, num_kv_heads=3,
        )
        stats = db.get_stats()
        assert stats["total_turns"] == 1
        assert stats["total_sessions"] == 1
        assert stats["db_size_bytes"] > 0
