#!/usr/bin/env python3
"""
CLI script to initialize Beverly Knits ERP AI Agent System
Equivalent to 'codebuff -init-agents' command
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ai_agents.agent_initializer import AgentSystemInitializer, initialize_ai_agents
except ImportError as e:
    print(f"Error importing agent system: {e}")
    print("Make sure you're in the project root directory and dependencies are installed")
    sys.exit(1)


def print_banner():
    """Print initialization banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                Beverly Knits ERP v2                           ║
║              AI Agent System Initializer                      ║
╚═══════════════════════════════════════════════════════════════╝
""")


async def init_agents(mode: str = "full", config_path: str = None):
    """Initialize the AI agent system"""
    print_banner()
    
    try:
        print(f"🚀 Starting AI agent initialization (mode: {mode})...\n")
        
        # Initialize the system
        status = await initialize_ai_agents(config_path, mode)
        
        # Print results
        if status['is_ready']:
            print("✅ AI Agent System Initialization SUCCESSFUL!\n")
            
            print("📊 Initialization Summary:")
            print(f"   • Components Initialized: {len(status['components_initialized'])}")
            print(f"   • Agents Deployed: {len(status['agents_deployed'])}")
            print(f"   • Started: {status['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • Completed: {status.get('completed_at', 'N/A')}")
            
            if status['components_initialized']:
                print("\n🔧 Components Initialized:")
                for component in status['components_initialized']:
                    print(f"   ✓ {component.replace('_', ' ').title()}")
            
            if status['agents_deployed']:
                print("\n🤖 Agents Deployed:")
                for agent_id in status['agents_deployed']:
                    print(f"   ✓ {agent_id}")
            
            print("\n🎉 The AI agent system is now ready for use!")
            print("\n📋 Next Steps:")
            print("   1. Access the ERP dashboard to view agent status")
            print("   2. Monitor agent performance through the API endpoints")
            print("   3. Begin agent training if in development mode")
            print("   4. Configure customer-specific agent stacks as needed")
            
        else:
            print("❌ AI Agent System Initialization FAILED!\n")
            
            if status['errors']:
                print("🚨 Errors encountered:")
                for i, error in enumerate(status['errors'], 1):
                    print(f"   {i}. {error}")
            
            print("\n🔧 Troubleshooting:")
            print("   1. Check that all dependencies are installed")
            print("   2. Verify configuration files are present")
            print("   3. Ensure database connections are available")
            print("   4. Check system logs for detailed error information")
            
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed with error: {str(e)}")
        print("\n🔧 Please check:")
        print("   • All required dependencies are installed")
        print("   • Configuration files are present and valid")
        print("   • System has sufficient resources")
        return False


async def status_check():
    """Check the status of the AI agent system"""
    print("🔍 Checking AI Agent System Status...\n")
    
    try:
        initializer = AgentSystemInitializer()
        status = await initializer.get_system_status()
        
        print("📊 System Status:")
        init_status = status.get('initialization_status', {})
        
        if init_status.get('is_ready'):
            print("   ✅ System Status: READY")
        else:
            print("   ⚠️  System Status: NOT READY")
        
        components = status.get('components', {})
        if components:
            print("\n🔧 Components:")
            for component, component_status in components.items():
                if isinstance(component_status, str):
                    print(f"   ✓ {component.replace('_', ' ').title()}: {component_status}")
                elif isinstance(component_status, dict):
                    print(f"   ✓ {component.replace('_', ' ').title()}: Active")
        
        agents_deployed = init_status.get('agents_deployed', [])
        if agents_deployed:
            print(f"\n🤖 Active Agents: {len(agents_deployed)}")
            for agent_id in agents_deployed:
                print(f"   • {agent_id}")
        else:
            print("\n🤖 No agents currently deployed")
        
        return True
        
    except Exception as e:
        print(f"❌ Status check failed: {str(e)}")
        return False


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Initialize Beverly Knits ERP AI Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_agents.py                    # Full initialization
  python init_agents.py --mode minimal     # Minimal setup for development
  python init_agents.py --status           # Check system status
  python init_agents.py --config custom.json # Use custom config
"""
    )
    
    parser.add_argument(
        "--mode", 
        choices=["full", "minimal", "training_only"],
        default="full",
        help="Initialization mode (default: full)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check system status instead of initializing"
    )
    
    args = parser.parse_args()
    
    # Run the appropriate function
    if args.status:
        success = asyncio.run(status_check())
    else:
        success = asyncio.run(init_agents(args.mode, args.config))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
