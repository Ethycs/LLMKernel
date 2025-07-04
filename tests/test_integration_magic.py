#!/usr/bin/env python3
"""
Integration tests for magic commands using a real kernel instance.

These tests actually execute magic commands in a kernel environment.
Run with: pytest tests/test_integration_magic.py -v
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_kernel.kernel import LLMKernel


@pytest.fixture
def kernel_instance():
    """Create a real kernel instance for integration testing."""
    # Set minimal environment
    os.environ['LLM_KERNEL_LOGGING'] = 'false'  # Disable logging for tests
    
    # Create kernel
    kernel = LLMKernel()
    
    # Override some settings for testing
    kernel.active_model = 'gpt-4o' if 'gpt-4o' in kernel.llm_clients else list(kernel.llm_clients.keys())[0] if kernel.llm_clients else None
    
    yield kernel
    
    # Cleanup
    kernel.do_shutdown(False)


@pytest.mark.integration
class TestMagicCommandsIntegration:
    """Integration tests for all magic commands."""
    
    def execute_magic(self, kernel, magic_cmd):
        """Execute a magic command and return result."""
        # Magic commands are handled in do_execute
        result = kernel.do_execute(
            magic_cmd,
            silent=False,
            store_history=True,
            user_expressions={},
            allow_stdin=False
        )
        return result
    
    def test_model_commands(self, kernel_instance):
        """Test model-related magic commands."""
        # List models
        result = self.execute_magic(kernel_instance, "%llm_models")
        assert result['status'] == 'ok'
        
        # Show current model
        result = self.execute_magic(kernel_instance, "%llm_model")
        assert result['status'] == 'ok'
        
        # Switch model (if multiple available)
        if len(kernel_instance.llm_clients) > 1:
            models = list(kernel_instance.llm_clients.keys())
            new_model = models[1] if kernel_instance.active_model == models[0] else models[0]
            result = self.execute_magic(kernel_instance, f"%llm_model {new_model}")
            assert result['status'] == 'ok'
            assert kernel_instance.active_model == new_model
    
    def test_chat_mode(self, kernel_instance):
        """Test chat mode toggling."""
        # Enable chat mode
        result = self.execute_magic(kernel_instance, "%llm_chat on")
        assert result['status'] == 'ok'
        assert kernel_instance.chat_mode == True
        
        # Check status
        result = self.execute_magic(kernel_instance, "%llm_chat status")
        assert result['status'] == 'ok'
        
        # Disable chat mode
        result = self.execute_magic(kernel_instance, "%llm_chat off")
        assert result['status'] == 'ok'
        assert kernel_instance.chat_mode == False
    
    def test_context_management(self, kernel_instance):
        """Test context management commands."""
        # Show status
        result = self.execute_magic(kernel_instance, "%llm_status")
        assert result['status'] == 'ok'
        
        # Pin a cell
        result = self.execute_magic(kernel_instance, "%llm_pin_cell 1")
        assert result['status'] == 'ok'
        assert 1 in kernel_instance.pinned_cells
        
        # Show pinned cells
        result = self.execute_magic(kernel_instance, "%llm_pin_cell")
        assert result['status'] == 'ok'
        
        # Unpin cell
        result = self.execute_magic(kernel_instance, "%llm_unpin_cell 1")
        assert result['status'] == 'ok'
        assert 1 not in kernel_instance.pinned_cells
        
        # Set context strategy
        result = self.execute_magic(kernel_instance, "%llm_context smart")
        assert result['status'] == 'ok'
    
    def test_conversation_management(self, kernel_instance):
        """Test conversation management commands."""
        # Add some history
        kernel_instance.conversation_history = [
            {"role": "user", "content": "Test message"}
        ]
        
        # Show history
        result = self.execute_magic(kernel_instance, "%llm_history")
        assert result['status'] == 'ok'
        
        # Clear history
        result = self.execute_magic(kernel_instance, "%llm_clear")
        assert result['status'] == 'ok'
        assert len(kernel_instance.conversation_history) == 0
    
    def test_export_import_context(self, kernel_instance):
        """Test context export and import."""
        # Setup test data
        kernel_instance.conversation_history = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"}
        ]
        kernel_instance.pinned_cells.add(1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Export context
            result = self.execute_magic(kernel_instance, f"%llm_export_context {temp_file}")
            assert result['status'] == 'ok'
            assert os.path.exists(temp_file)
            
            # Clear state
            kernel_instance.conversation_history = []
            kernel_instance.pinned_cells.clear()
            
            # Import context
            result = self.execute_magic(kernel_instance, f"%llm_import_context {temp_file}")
            assert result['status'] == 'ok'
            
            # Verify imported
            assert len(kernel_instance.conversation_history) == 2
            assert 1 in kernel_instance.pinned_cells
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_info_commands(self, kernel_instance):
        """Test informational commands."""
        # Kernel info
        result = self.execute_magic(kernel_instance, "%llm_info")
        assert result['status'] == 'ok'
        
        # Token usage
        result = self.execute_magic(kernel_instance, "%llm_tokens")
        assert result['status'] == 'ok'
        
        # Cost tracking
        result = self.execute_magic(kernel_instance, "%llm_cost")
        assert result['status'] == 'ok'
    
    def test_configuration(self, kernel_instance):
        """Test configuration commands."""
        # Show config (this creates widgets, so might not display in tests)
        result = self.execute_magic(kernel_instance, "%llm_config")
        assert result['status'] == 'ok'
    
    def test_reset_command(self, kernel_instance):
        """Test kernel reset."""
        # Add some state
        kernel_instance.conversation_history = [{"test": "data"}]
        kernel_instance.pinned_cells.add(1)
        kernel_instance.context_cells = [1, 2, 3]
        
        # Reset
        result = self.execute_magic(kernel_instance, "%llm_reset")
        assert result['status'] == 'ok'
        
        # Verify reset
        assert len(kernel_instance.conversation_history) == 0
        assert len(kernel_instance.pinned_cells) == 0
        assert len(kernel_instance.context_cells) == 0
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="No OpenAI API key")
    def test_llm_query(self, kernel_instance):
        """Test actual LLM query (requires API key)."""
        # Skip if no models available
        if not kernel_instance.llm_clients:
            pytest.skip("No LLM clients configured")
        
        # Simple query using cell magic
        code = """%%llm
What is 2 + 2?"""
        
        result = kernel_instance.do_execute(
            code,
            silent=False,
            store_history=True,
            user_expressions={},
            allow_stdin=False
        )
        
        assert result['status'] == 'ok'
        
        # Verify response was tracked
        assert len(kernel_instance.session_exchanges) > 0
    
    def test_multimodal_commands(self, kernel_instance):
        """Test multimodal commands (without actual files)."""
        # List files (should be empty)
        result = self.execute_magic(kernel_instance, "%llm_files_list")
        assert result['status'] == 'ok'
        
        # Cache info
        if hasattr(kernel_instance, 'cache_manager'):
            result = self.execute_magic(kernel_instance, "%llm_cache_info")
            assert result['status'] == 'ok'
            
            result = self.execute_magic(kernel_instance, "%llm_cache_list")
            assert result['status'] == 'ok'
    
    def test_mcp_commands(self, kernel_instance):
        """Test MCP commands."""
        # List MCP servers
        result = self.execute_magic(kernel_instance, "%llm_mcp_list")
        assert result['status'] == 'ok'
        
        # MCP status
        result = self.execute_magic(kernel_instance, "%llm_mcp_status")
        assert result['status'] == 'ok'


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations."""
    
    async def test_async_llm_query(self, kernel_instance):
        """Test async LLM query."""
        if not kernel_instance.llm_clients:
            pytest.skip("No LLM clients configured")
        
        # Create a simple query
        query = "What is 2 + 2?"
        
        # Execute async query
        from llm_kernel.llm_integration import LLMIntegration
        llm = LLMIntegration(kernel_instance)
        
        result = await llm.query_llm_async(query)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


# Additional test utilities
def run_magic_command_test(kernel, command, expected_status='ok'):
    """Helper to run a magic command and check status."""
    result = kernel.do_execute(
        command,
        silent=False,
        store_history=True,
        user_expressions={},
        allow_stdin=False
    )
    assert result['status'] == expected_status
    return result


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])