#!/usr/bin/env python3
"""
Startup script for Beverly Knits ERP AI Agent System
Simple wrapper around the agent initializer
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ai_agents.agent_initializer import AgentSystemInitializer
except ImportError as e:
    print(f"❌ Cannot import AI agent system: {e}")
    print("Make sure you're in the Beverly Knits ERP project directory")
    sys.exit(1)


async def main():
    """Main startup function"""
    print("🚀 Starting Beverly Knits ERP AI Agent System...")
    
    # Initialize the agent system
    initializer = AgentSystemInitializer()
    
    try:
        # Initialize in full mode
        status = await initializer.initialize_system(mode="full")
        
        if status['is_ready']:
            print("\n✅ AI Agent System is running!")
            print(f"📊 {len(status['agents_deployed'])} agents deployed")
            print(f"🔧 {len(status['components_initialized'])} components active")
            
            print("\n🎯 System ready for operation. Press Ctrl+C to shutdown.")
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                await initializer.shutdown_system()
                print("✅ AI Agent System stopped.")
        else:
            print("\n❌ AI Agent System failed to start.")
            if status.get('errors'):
                for error in status['errors']:
                    print(f"   • {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
