#!/usr/bin/env python3
"""
Test script to verify kernel imports and initialization work correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        print("  - Importing kernel module...", end=" ")
        from llm_kernel.kernel import LLMKernel
        print("✓")
        
        print("  - Importing magic command modules...", end=" ")
        from llm_kernel.magic_commands import (
            BaseMagics, ContextMagics, MCPMagics, 
            RerankingMagics, ConfigMagics
        )
        print("✓")
        
        print("  - Importing support modules...", end=" ")
        from llm_kernel.context_manager import ContextManager, ExecutionTracker
        from llm_kernel.dialogue_pruner import DialoguePruner
        from llm_kernel.config_manager import ConfigManager
        from llm_kernel.llm_integration import LLMIntegration
        print("✓")
        
        print("  - Checking optional imports...", end=" ")
        try:
            from llm_kernel.mcp_manager import MCPManager
            print("✓ (MCP available)")
        except ImportError:
            print("⚠ (MCP not available - fastmcp not installed)")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_kernel_creation():
    """Test that kernel can be created."""
    print("\nTesting kernel creation...")
    
    try:
        # Mock IPython shell for testing
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        from IPython.core.profiledir import ProfileDir
        
        print("  - Creating mock shell...", end=" ")
        profile_dir = ProfileDir.create_profile_dir_by_name('.', 'default')
        shell = TerminalInteractiveShell.instance(profile_dir=profile_dir)
        print("✓")
        
        print("  - Creating kernel instance...", end=" ")
        from llm_kernel.kernel import LLMKernel
        
        # Create minimal kernel args
        kernel = LLMKernel(
            session=None,
            shell_socket=None,
            iopub_socket=None,
            stdin_socket=None,
            control_socket=None,
            hb_socket=None,
            shell=shell,
            user_module=None,
            user_ns=None,
            _show_traceback=True
        )
        print("✓")
        
        print("  - Checking kernel attributes...", end=" ")
        assert hasattr(kernel, 'llm_clients')
        assert hasattr(kernel, 'conversation_history')
        assert hasattr(kernel, 'context_manager')
        print("✓")
        
        print("\n✅ Kernel creation successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Kernel creation error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== LLM Kernel Test Suite ===\n")
    
    success = True
    success &= test_imports()
    
    # Only test kernel creation if imports succeeded
    if success:
        success &= test_kernel_creation()
    
    print("\n" + "="*30)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()