#!/usr/bin/env python3
"""
Test script for the new YAML-based CrewAI configuration.
"""

import sys
from pathlib import Path

# Add src to path
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

def test_yaml_crew():
    """Test the YAML-based crew configuration."""
    try:
        from src.agents.crew import MultiClassRagCrew, kickoff_query_yaml
        print("✓ Successfully imported MultiClassRagCrew and kickoff_query_yaml")
        
        # Test crew instantiation
        crew_instance = MultiClassRagCrew()
        print("✓ Successfully created MultiClassRagCrew instance")
        
        # Test agents access
        agents = ['policy', 'research', 'attack', 'security', 'ai', 'training', 'general']
        for agent_name in agents:
            agent = getattr(crew_instance, agent_name)()
            print(f"✓ Successfully created {agent_name} agent: {agent.role}")
        
        # Test crew creation
        crew = crew_instance.crew()
        print("✓ Successfully created crew with all agents")
        
        print("\n🎉 All tests passed! YAML configuration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_legacy_compatibility():
    """Test that legacy functions still work."""
    try:
        from src.agents.crew_setup import kickoff_query, build_agents
        print("✓ Successfully imported legacy functions")
        
        # Test agents building
        agents = build_agents()
        print(f"✓ Successfully built {len(agents)} legacy agents")
        
        print("✓ Legacy compatibility maintained")
        return True
        
    except Exception as e:
        print(f"❌ Legacy test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing YAML-based CrewAI configuration...\n")
    
    yaml_success = test_yaml_crew()
    print("\n" + "="*50 + "\n")
    legacy_success = test_legacy_compatibility()
    
    print("\n" + "="*50)
    if yaml_success and legacy_success:
        print("🎉 All tests passed! Both YAML and legacy approaches work.")
    else:
        print("❌ Some tests failed. Check the errors above.")
