#!/bin/bash
# CLI Benchmark: Builtin FTS5 vs kv-memory via actual Hermes CLI
#
# Requires: DEEPSEEK_API_KEY set, hermes CLI in PATH or repo root
# Usage: bash benchmarks/memory/cli_benchmark.sh

set +e  # Don't abort on non-zero exit (hermes may crash on shutdown)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HERMES="$REPO_ROOT/.venv/bin/python $REPO_ROOT/hermes"
YOLO="--yolo --cli"
RESULTS_FILE="benchmarks/memory/results/cli_bench_results.txt"

# Clean up old state
rm -rf /tmp/hermes_fts5_home /tmp/hermes_kv_home
mkdir -p /tmp/hermes_fts5_home /tmp/hermes_kv_home benchmarks/memory/results

echo "=============================================" | tee "$RESULTS_FILE"
echo "CLI Benchmark: Builtin FTS5 vs kv-memory" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# ── Test 1: Builtin FTS5 ──────────────────────────────────────────

echo "### Test 1: Builtin FTS5 Memory ###" | tee -a "$RESULTS_FILE"

# Configure FTS5 home (no external memory provider)
cat > /tmp/hermes_fts5_home/config.yaml << 'EOF'
model:
  provider: deepseek
  model: deepseek-chat
memory:
  provider: ""
EOF

echo "  Teaching fact via builtin memory..." | tee -a "$RESULTS_FILE"
HERMES_HOME=/tmp/hermes_fts5_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 60 $HERMES -z \
  "My production API key is: sk-live-abc123def456. My staging server IP is 10.0.0.55. My preferred deployment tool is Docker Compose. Please store ALL of these in memory." \
  --resume cli-bench-fts5 $YOLO 2>&1 | tail -3 | tee -a "$RESULTS_FILE"

sleep 2

echo "  Query 1: Exact keyword match..." | tee -a "$RESULTS_FILE"
FTS5_Q1=$(HERMES_HOME=/tmp/hermes_fts5_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 45 $HERMES -z \
  "What is my API key? Search memory to find it." \
  --resume cli-bench-fts5 $YOLO 2>&1)
FTS5_HIT1=$(echo "$FTS5_Q1" | grep -c "abc123def456\|sk-live" || echo "0")
echo "  FTS5 API key found: $FTS5_HIT1" | tee -a "$RESULTS_FILE"

sleep 2

echo "  Query 2: Semantic (different words)..." | tee -a "$RESULTS_FILE"
FTS5_Q2=$(HERMES_HOME=/tmp/hermes_fts5_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 45 $HERMES -z \
  "What IP address does our test environment run on? Search memory." \
  --resume cli-bench-fts5 $YOLO 2>&1)
FTS5_HIT2=$(echo "$FTS5_Q2" | grep -c "10.0.0.55" || echo "0")
echo "  FTS5 IP found: $FTS5_HIT2" | tee -a "$RESULTS_FILE"

sleep 2

echo "  Query 3: Preference recall..." | tee -a "$RESULTS_FILE"
FTS5_Q3=$(HERMES_HOME=/tmp/hermes_fts5_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 45 $HERMES -z \
  "How do I deploy my applications? What tool do I use?" \
  --resume cli-bench-fts5 $YOLO 2>&1)
FTS5_HIT3=$(echo "$FTS5_Q3" | grep -c -i "docker" || echo "0")
echo "  FTS5 Docker found: $FTS5_HIT3" | tee -a "$RESULTS_FILE"

# ── Test 2: kv-memory Provider ─────────────────────────────────────

echo "" | tee -a "$RESULTS_FILE"
echo "### Test 2: kv-memory Provider ###" | tee -a "$RESULTS_FILE"

# Configure kv-memory home
cat > /tmp/hermes_kv_home/config.yaml << 'EOF'
model:
  provider: deepseek
  model: deepseek-chat
memory:
  provider: kv_memory
plugins:
  kv-memory:
    embedding_backend: sentence-transformers
    top_k: 5
    storage_mode: fp16
    diversity_lambda: 1.0
EOF

echo "  Teaching fact via kv-memory..." | tee -a "$RESULTS_FILE"
HERMES_HOME=/tmp/hermes_kv_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 60 $HERMES -z \
  "My production API key is: sk-live-abc123def456. My staging server IP is 10.0.0.55. My preferred deployment tool is Docker Compose. Please store ALL of these using kv_memory_search or the builtin memory tool." \
  --resume cli-bench-kv $YOLO 2>&1 | tail -3 | tee -a "$RESULTS_FILE"

sleep 2

echo "  Query 1: Exact keyword match..." | tee -a "$RESULTS_FILE"
KV_Q1=$(HERMES_HOME=/tmp/hermes_kv_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 45 $HERMES -z \
  "Use kv_memory_search to find my API key. What is it?" \
  --resume cli-bench-kv $YOLO 2>&1)
KV_HIT1=$(echo "$KV_Q1" | grep -c "abc123def456\|sk-live" || echo "0")
echo "  kv-memory API key found: $KV_HIT1" | tee -a "$RESULTS_FILE"

sleep 2

echo "  Query 2: Semantic (different words)..." | tee -a "$RESULTS_FILE"
KV_Q2=$(HERMES_HOME=/tmp/hermes_kv_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 45 $HERMES -z \
  "Use kv_memory_search to find: what is the IP of our staging server?" \
  --resume cli-bench-kv $YOLO 2>&1)
KV_HIT2=$(echo "$KV_Q2" | grep -c "10.0.0.55" || echo "0")
echo "  kv-memory IP found: $KV_HIT2" | tee -a "$RESULTS_FILE"

sleep 2

echo "  Query 3: Preference recall..." | tee -a "$RESULTS_FILE"
KV_Q3=$(HERMES_HOME=/tmp/hermes_kv_home DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  timeout 45 $HERMES -z \
  "Use kv_memory_search to find: how do I deploy my applications?" \
  --resume cli-bench-kv $YOLO 2>&1)
KV_HIT3=$(echo "$KV_Q3" | grep -c -i "docker" || echo "0")
echo "  kv-memory Docker found: $KV_HIT3" | tee -a "$RESULTS_FILE"

# ── Summary ────────────────────────────────────────────────────────

echo "" | tee -a "$RESULTS_FILE"
echo "### Results ###" | tee -a "$RESULTS_FILE"
echo "FTS5:   API=$FTS5_HIT1  IP=$FTS5_HIT2  Docker=$FTS5_HIT3  (Total: $((FTS5_HIT1 + FTS5_HIT2 + FTS5_HIT3))/3)" | tee -a "$RESULTS_FILE"
echo "kv-mem: API=$KV_HIT1  IP=$KV_HIT2  Docker=$KV_HIT3  (Total: $((KV_HIT1 + KV_HIT2 + KV_HIT3))/3)" | tee -a "$RESULTS_FILE"
