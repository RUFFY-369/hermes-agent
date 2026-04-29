import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evolution.tinker import TinkerBridgeTrainer

async def test_tinker_handshake():
    print("🚀 PHASE 3: Testing Tinker API Bridge & Handshake")
    print("=" * 60)

    # 1. Check API Key
    api_key = os.getenv("TINKER_API_KEY")
    
    if not api_key:
        print("⚠️  TINKER_API_KEY not found in environment.")
        print("   Testing graceful fallback mode...")
        
        # 2. Verify Graceful Fallback
        bridge = TinkerBridgeTrainer(use_tinker=True)
        
        if not bridge.is_active():
            print("⚠️  TINKER_API_KEY missing. Falling back to local training.")
            print("✅ Graceful fallback verified — bridge correctly deactivated.")
            print("\n✅ SUCCESS: Tinker Bridge graceful fallback works correctly.")
        else:
            print("❌ ERROR: Bridge should NOT be active without API key!")
        return

    # 3. Initialize Bridge with API key present
    bridge = TinkerBridgeTrainer(use_tinker=True)
    
    if not bridge.is_active():
        print("❌ ERROR: Tinker Bridge could not be activated.")
        print("Verify that tools.rl_training_tool is accessible.")
        return

    print("📡 [Bridge] Sending test heartbeat to Tinker...")
    
    # 4. Simulate a Training Request Handshake
    try:
        from tools.rl_training_tool import rl_get_current_config
        config = await rl_get_current_config()
        print(f"✅ Connection Verified. Tinker Config Loaded: {json.loads(config).get('environment', 'unknown')}")
        print("\n✅ SUCCESS: Tinker Bridge is ready for production offloading.")
    except Exception as e:
        print(f"❌ Handshake Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tinker_handshake())
