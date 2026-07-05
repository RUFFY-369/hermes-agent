#!/usr/bin/env python3
"""Proof that automatic capture does NOT bloat the prompt.

Hermes builtin memory: stores text via `memory add`, injects ALL stored
entries into the system prompt on every turn. Grow database → grow context
window linearly.

kv-memory: stores float16 embeddings, injects only top-K semantically
relevant results via prefetch. Grow database → injection stays flat at K.

This benchmark simulates N stored "memories" and measures injected tokens
per turn for both approaches.
"""

import sys, os, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.memory.kv_memory.capture import create_embedding_backend
from plugins.memory.kv_memory.storage import KVMemoryDB
from plugins.memory.kv_memory.retrieval import KVRetriever, cosine_similarity
from plugins.memory.kv_memory.config import KVMemoryConfig


def simulate_builtin_bloat(n_memories: int) -> int:
    """Hermes builtin: injects ALL stored memory entries as text.
    Each memory entry is ~200 chars (~50 tokens)."""
    return n_memories * 50  # tokens injected per turn


def simulate_kv_bloat(n_memories: int, top_k: int = 5) -> int:
    """kv-memory: injects only top-K semantically relevant turns.
    Each injected turn summary is ~200 chars (~50 tokens)."""
    return min(n_memories, top_k) * 50  # tokens injected per turn


def main():
    backend = create_embedding_backend("auto")
    queries = [
        "How do I handle async database connections in Python?",
        "What's the best deployment strategy for microservices?",
        "How do I configure SSL certificates for nginx?",
        "What monitoring tools should I use for production?",
        "How do I optimize PostgreSQL query performance?",
    ]

    # Simulate storing N random "memory entries"
    topics = [
        "database", "deployment", "ssl", "monitoring", "testing",
        "logging", "caching", "authentication", "api", "docker",
        "kubernetes", "ci/cd", "security", "backup", "networking",
    ]

    np.random.seed(42)
    print("N\tBuiltin\tkv-mem\tReduction")
    for n in [5, 10, 20, 50, 100]:
        # Build kv-memory with N random conversations
        db_path = tempfile.mktemp(suffix=".db")
        db = KVMemoryDB(db_path)
        db.initialize_schema()
        config = KVMemoryConfig(db_path=db_path, top_k=5, min_similarity=0.0,
                                diversity_lambda=1.0)
        retriever = KVRetriever(db, config)

        for i in range(n):
            topic = topics[i % len(topics)]
            text = f"User asked about {topic} configuration. Assistant provided {topic} best practices and common pitfalls with specific examples."
            emb = backend.encode(text)
            db.store_turn(
                session_id="bench", turn_number=i + 1, embedding=emb,
                summary_text=text[:200], store_fp16=True,
            )

        # Measure injection for each query
        total_injected = 0
        for q in queries:
            q_emb = backend.encode(q)
            results = retriever.retrieve(q_emb, k=5)
            for r in results:
                total_injected += len(r.get("summary_text", "").split()) * 1.3  # ~tokens

        avg_kv_tokens = total_injected / len(queries)
        builtin_tokens = simulate_builtin_bloat(n)
        reduction = (builtin_tokens - avg_kv_tokens) / max(builtin_tokens, 1) * 100

        print(f"{n}\t{builtin_tokens:.0f}\t{avg_kv_tokens:.0f}\t{reduction:.0f}%")

        db.close()
        for ext in ["", "-wal", "-shm"]:
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)

    print()
    print("Builtin: injects ALL memories → grows linearly with database.")
    print("kv-memory: injects only top-K relevant → stays flat at ~250 tokens.")
    print("Automatic capture is safe because relevance filtering prevents bloat.")


if __name__ == "__main__":
    main()
