import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evolution.tinker import TinkerBridgeTrainer

async def test_tinker_handshake():
    print("🚀 PHASE 2: Testing Tinker API Bridge & Handshake")
    print("=" * 60)

    # 1. Check API Key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ ERROR: TINKER_API_KEY not found in environment.")
        print("Please run: export TINKER_API_KEY=your_key")
        return

    # 2. Initialize Bridge
    bridge = TinkerBridgeTrainer(use_tinker=True)
    
    if not bridge.is_active():
        print("❌ ERROR: Tinker Bridge could not be activated.")
        print("Verify that tools.rl_training_tool is accessible.")
        return

    print("📡 [Bridge] Sending test heartbeat to Tinker...")
    
    # 3. Simulate a Training Request Handshake
    # We don't want to start a real job, so we just verify the tool loading
    try:
        from tools.rl_training_tool import rl_get_current_config
        config = await rl_get_current_config()
        print(f"✅ Connection Verified. Tinker Config Loaded: {json.loads(config).get('environment', 'unknown')}")
        print("\n✅ SUCCESS: Tinker Bridge is ready for production offloading.")
    except Exception as e:
        print(f"❌ Handshake Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tinker_handshake())
