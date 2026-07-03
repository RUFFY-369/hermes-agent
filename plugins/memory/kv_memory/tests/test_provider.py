"""Integration tests for KVMemoryProvider."""

import json
import os
import tempfile

import pytest

from plugins.memory.kv_memory.config import KVMemoryConfig
from plugins.memory.kv_memory.provider import (
    KVMemoryProvider,
    KV_MEMORY_SEARCH_SCHEMA,
    KV_MEMORY_STATUS_SCHEMA,
)


@pytest.fixture
def provider():
    """Create a provider with a temporary database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    config = KVMemoryConfig(
        db_path=path,
        top_k=3,
        min_similarity=0.0,
        temporal_decay_half_life=0,
        retention_days=0,
    )
    p = KVMemoryProvider(config=config)
    yield p
    p.shutdown()
    for ext in ["", "-wal", "-shm"]:
        f = path + ext
        if os.path.exists(f):
            os.unlink(f)


class TestProviderInterface:
    """Test MemoryProvider ABC compliance."""

    def test_name(self, provider):
        assert provider.name == "kv-memory"

    def test_is_available(self, provider):
        assert provider.is_available() is True

    def test_initialize(self, provider):
        provider.initialize("test-session", platform="test", model="test-model")
        assert provider._initialized is True
        assert provider._session_id == "test-session"

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "kv_memory_search" in names
        assert "kv_memory_status" in names

    def test_system_prompt_block_empty(self, provider):
        provider.initialize("test", platform="test")
        block = provider.system_prompt_block()
        assert "KV" in block
        assert "Empty" in block or "0 turns" in block.lower()

    def test_system_prompt_block_populated(self, provider):
        provider.initialize("test", platform="test")
        # Store a turn to populate
        provider.sync_turn("Hello", "Hi there", session_id="test")
        block = provider.system_prompt_block()
        assert "KV" in block
        assert "1 turns" in block or "1 turn" in block


class TestSyncAndPrefetch:
    """Turn storage and retrieval."""

    def test_sync_turn(self, provider):
        provider.initialize("test", platform="test")
        provider.sync_turn(
            user_content="How does Python asyncio work?",
            assistant_content="The event loop manages coroutines cooperatively.",
            session_id="test",
        )
        assert provider._turn_number == 1
        stats = json.loads(provider.handle_tool_call("kv_memory_status", {}))
        assert stats["total_turns"] == 1
        assert stats["total_sessions"] == 1
        assert stats["db_size_mb"] >= 0

    def test_sync_multiple_turns(self, provider):
        provider.initialize("test", platform="test")
        for i in range(5):
            provider.sync_turn(f"Question {i}", f"Answer {i}", session_id="test")
        stats = json.loads(provider.handle_tool_call("kv_memory_status", {}))
        assert stats["total_turns"] == 5

    def test_prefetch_returns_context(self, provider):
        provider.initialize("test", platform="test")
        provider.sync_turn("Async question", "Async answer", session_id="test")
        provider.queue_prefetch("async programming")
        context = provider.prefetch("async programming")
        # May return empty if similarity too low, but shouldn't crash
        assert isinstance(context, str)

    def test_queue_prefetch_empty_query(self, provider):
        provider.initialize("test", platform="test")
        provider.queue_prefetch("")  # Should not crash
        context = provider.prefetch("")
        assert context == ""


class TestToolHandlers:
    """Tool call handling."""

    def test_search(self, provider):
        provider.initialize("test", platform="test")
        provider.sync_turn("DB password is xyz123", "Got it", session_id="test")
        result = json.loads(provider.handle_tool_call("kv_memory_search", {"query": "password"}))
        assert "results" in result
        assert "count" in result
        assert "backend" in result

    def test_search_no_query(self, provider):
        provider.initialize("test", platform="test")
        result = json.loads(provider.handle_tool_call("kv_memory_search", {}))
        assert "error" in result

    def test_search_custom_limit(self, provider):
        provider.initialize("test", platform="test")
        for i in range(5):
            provider.sync_turn(f"Question {i}", f"Answer {i}", session_id="test")
        result = json.loads(provider.handle_tool_call("kv_memory_search",
                            {"query": "question", "limit": 2}))
        assert result["count"] <= 2

    def test_status(self, provider):
        provider.initialize("test", platform="test")
        result = json.loads(provider.handle_tool_call("kv_memory_status", {}))
        assert "total_turns" in result
        assert "total_sessions" in result
        assert "backend" in result
        assert "embedding_dim" in result

    def test_unknown_tool(self, provider):
        provider.initialize("test", platform="test")
        with pytest.raises(NotImplementedError):
            provider.handle_tool_call("nonexistent_tool", {})


class TestHooks:
    """Optional lifecycle hooks."""

    def test_on_session_end(self, provider):
        provider.initialize("test", platform="test")
        provider.sync_turn("Q", "A", session_id="test")
        # Should not crash
        provider.on_session_end([{"role": "user", "content": "Q"}])

    def test_on_memory_write(self, provider):
        provider.initialize("test", platform="test")
        provider.on_memory_write("add", "memory", "Remember this fact")
        stats = json.loads(provider.handle_tool_call("kv_memory_status", {}))
        assert stats["total_turns"] >= 1  # Should have synced the write

    def test_on_session_switch_reset(self, provider):
        provider.initialize("s1", platform="test")
        provider.sync_turn("Q1", "A1", session_id="s1")
        provider.on_session_switch("s2", parent_session_id="s1", reset=True)
        assert provider._turn_number == 0  # Counter reset
        assert provider._session_id == "s2"

    def test_shutdown_cleanup(self, provider):
        provider.initialize("test", platform="test")
        provider.shutdown()
        assert provider._initialized is False
        assert provider._db is None
        # Double shutdown should not crash
        provider.shutdown()


class TestConfigIntegration:
    """Config schema and save."""

    def test_get_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert len(schema) > 0
        keys = [f["key"] for f in schema]
        assert "embedding_backend" in keys
        assert "top_k" in keys

    def test_save_config(self, provider):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            provider.save_config({"top_k": "10"}, d)
            config_path = os.path.join(d, "config.yaml")
            assert os.path.exists(config_path)
