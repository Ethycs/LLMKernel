#!/usr/bin/env python3
"""
Pytest integration for testing all LLM Kernel magic commands.

Run with:
    pytest tests/test_magic_commands.py -v
    pytest tests/test_magic_commands.py -k "specific_test"
    pytest tests/test_magic_commands.py --tb=short  # shorter traceback
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_kernel.kernel import LLMKernel
from llm_kernel.magic_commands.base import BaseMagics
from llm_kernel.magic_commands.context import ContextMagics
from llm_kernel.magic_commands.models import ModelMagics
from llm_kernel.magic_commands.config import ConfigMagics
from llm_kernel.magic_commands.mcp import MCPMagics
from llm_kernel.magic_commands.multimodal import MultimodalMagics
from llm_kernel.magic_commands.multimodal_native_pdf import NativePDFMagics
from llm_kernel.magic_commands.cache import CacheMagics


@pytest.fixture
def mock_kernel():
    """Create a mock kernel instance for testing."""
    kernel = Mock(spec=LLMKernel)
    
    # Set up basic attributes
    kernel.llm_clients = {
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        'claude-3-opus': 'claude-3-opus-20240229',
        'gemini-2.5-pro': 'gemini/gemini-2.5-pro'
    }
    kernel.active_model = 'gpt-4o'
    kernel.conversation_history = []
    kernel.context_cells = []
    kernel.pinned_cells = set()
    kernel.chat_mode = False
    kernel.display_mode = 'inline'
    kernel.execution_count = 1
    kernel.session_exchanges = []
    kernel.log = Mock()
    
    # Mock methods
    kernel.get_notebook_cells_as_context = Mock(return_value=[])
    kernel.query_llm_async = Mock(return_value="Test response")
    kernel.track_exchange = Mock()
    kernel.update_context = Mock()
    
    # Mock file managers
    kernel.file_upload_manager = Mock()
    kernel.cache_manager = Mock()
    
    return kernel


@pytest.fixture
def mock_shell():
    """Create a mock IPython shell."""
    shell = Mock()
    shell.user_ns = {}
    return shell


class TestBaseMagics:
    """Test basic magic commands."""
    
    def test_llm_chat_toggle(self, mock_kernel, mock_shell):
        """Test chat mode toggling."""
        magics = BaseMagics(mock_shell, mock_kernel)
        
        # Test toggle on
        magics.llm_chat("")
        assert mock_kernel.chat_mode == True
        assert mock_kernel.display_mode == 'chat'
        
        # Test toggle off
        magics.llm_chat("")
        assert mock_kernel.chat_mode == False
        assert mock_kernel.display_mode == 'inline'
    
    def test_llm_chat_explicit(self, mock_kernel, mock_shell):
        """Test explicit chat mode settings."""
        magics = BaseMagics(mock_shell, mock_kernel)
        
        # Test on
        magics.llm_chat("on")
        assert mock_kernel.chat_mode == True
        
        # Test off
        magics.llm_chat("off")
        assert mock_kernel.chat_mode == False
        
        # Test status
        magics.llm_chat("status")  # Should just print, not change state
    
    def test_llm_clear(self, mock_kernel, mock_shell):
        """Test clearing conversation history."""
        magics = BaseMagics(mock_shell, mock_kernel)
        
        # Add some history
        mock_kernel.conversation_history = [{"role": "user", "content": "test"}]
        mock_kernel.context_cells = [1, 2, 3]
        
        # Clear
        magics.llm_clear("")
        
        # Verify cleared
        assert mock_kernel.conversation_history == []
        assert mock_kernel.context_cells == []
    
    def test_llm_history(self, mock_kernel, mock_shell):
        """Test showing conversation history."""
        magics = BaseMagics(mock_shell, mock_kernel)
        
        # Add some history
        mock_kernel.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # This should print the history
        magics.llm_history("")
    
    @pytest.mark.asyncio
    async def test_llm_query(self, mock_kernel, mock_shell):
        """Test basic LLM query."""
        magics = BaseMagics(mock_shell, mock_kernel)
        
        # Mock the async query
        mock_kernel.query_llm_async = Mock(return_value="Test response")
        
        # Test query
        result = magics.llm("", "What is 2+2?")
        
        # Should trigger the mock
        mock_kernel.query_llm_async.assert_called()


class TestModelMagics:
    """Test model management magic commands."""
    
    def test_llm_models(self, mock_kernel, mock_shell):
        """Test listing available models."""
        magics = ModelMagics(mock_shell, mock_kernel)
        
        # Should print available models
        magics.llm_models("")
    
    def test_llm_model_show_current(self, mock_kernel, mock_shell):
        """Test showing current model."""
        magics = ModelMagics(mock_shell, mock_kernel)
        
        # No argument should show current model
        magics.llm_model("")
    
    def test_llm_model_switch(self, mock_kernel, mock_shell):
        """Test switching models."""
        magics = ModelMagics(mock_shell, mock_kernel)
        
        # Switch to valid model
        magics.llm_model("gpt-4o-mini")
        assert mock_kernel.active_model == "gpt-4o-mini"
        
        # Try invalid model
        magics.llm_model("invalid-model")
        # Should still be the previous model
        assert mock_kernel.active_model == "gpt-4o-mini"


class TestContextMagics:
    """Test context management magic commands."""
    
    def test_llm_status(self, mock_kernel, mock_shell):
        """Test showing kernel status."""
        magics = ContextMagics(mock_shell, mock_kernel)
        
        # Should print status info
        magics.llm_status("")
    
    def test_llm_pin_cell(self, mock_kernel, mock_shell):
        """Test pinning cells."""
        magics = ContextMagics(mock_shell, mock_kernel)
        
        # Pin a cell
        magics.llm_pin_cell("5")
        assert 5 in mock_kernel.pinned_cells
        
        # Show pinned cells
        magics.llm_pin_cell("")
    
    def test_llm_unpin_cell(self, mock_kernel, mock_shell):
        """Test unpinning cells."""
        magics = ContextMagics(mock_shell, mock_kernel)
        
        # Setup
        mock_kernel.pinned_cells.add(5)
        
        # Unpin
        magics.llm_unpin_cell("5")
        assert 5 not in mock_kernel.pinned_cells
    
    def test_llm_context_strategy(self, mock_kernel, mock_shell):
        """Test setting context strategy."""
        magics = ContextMagics(mock_shell, mock_kernel)
        
        # Set strategy
        mock_kernel.context_strategy = 'chronological'
        magics.llm_context("smart")
        
        # Note: This would normally update kernel.context_strategy
        # but our mock doesn't implement the full logic
    
    def test_llm_tokens(self, mock_kernel, mock_shell):
        """Test showing token usage."""
        magics = ContextMagics(mock_shell, mock_kernel)
        
        # Mock token counting
        mock_kernel.count_tokens = Mock(return_value=100)
        mock_kernel.context_cells = [1, 2, 3]
        
        # Should print token info
        magics.llm_tokens("")


class TestConfigMagics:
    """Test configuration magic commands."""
    
    def test_llm_info(self, mock_kernel, mock_shell):
        """Test showing kernel info."""
        magics = ConfigMagics(mock_shell, mock_kernel)
        
        # Mock version and environment
        with patch('platform.python_version', return_value='3.10.0'):
            magics.llm_info("")
    
    def test_llm_cost(self, mock_kernel, mock_shell):
        """Test showing cost tracking."""
        magics = ConfigMagics(mock_shell, mock_kernel)
        
        # Mock session costs
        mock_kernel.session_costs = {
            'total': 0.05,
            'by_model': {
                'gpt-4o': 0.03,
                'gpt-4o-mini': 0.02
            }
        }
        
        # Should print cost info
        magics.llm_cost("")
    
    def test_llm_reset(self, mock_kernel, mock_shell):
        """Test resetting kernel state."""
        magics = ConfigMagics(mock_shell, mock_kernel)
        
        # Setup some state
        mock_kernel.conversation_history = [{"test": "data"}]
        mock_kernel.context_cells = [1, 2]
        mock_kernel.pinned_cells = {3, 4}
        
        # Reset
        magics.llm_reset("")
        
        # Verify reset
        assert mock_kernel.conversation_history == []
        assert mock_kernel.context_cells == []
        assert len(mock_kernel.pinned_cells) == 0


class TestMultimodalMagics:
    """Test multimodal magic commands."""
    
    def test_llm_image(self, mock_kernel, mock_shell):
        """Test image loading."""
        magics = MultimodalMagics(mock_shell, mock_kernel)
        
        # Mock multimodal handler
        mock_kernel.multimodal = Mock()
        mock_kernel.multimodal.add_image = Mock()
        
        # Test with non-existent file (should handle gracefully)
        magics.llm_image("nonexistent.png")
    
    def test_llm_paste(self, mock_kernel, mock_shell):
        """Test clipboard paste."""
        magics = MultimodalMagics(mock_shell, mock_kernel)
        
        # Mock multimodal handler
        mock_kernel.multimodal = Mock()
        
        # Test paste (will depend on clipboard content)
        with patch('llm_kernel.multimodal.MultimodalContent.get_clipboard_content', 
                   return_value={'type': 'text', 'data': 'test'}):
            magics.llm_paste("")


class TestPDFMagics:
    """Test PDF handling magic commands."""
    
    def test_llm_pdf_native(self, mock_kernel, mock_shell):
        """Test native PDF upload."""
        magics = NativePDFMagics(mock_shell, mock_kernel)
        
        # Test with non-existent file
        magics.llm_pdf_native("nonexistent.pdf")
    
    def test_llm_files_list(self, mock_kernel, mock_shell):
        """Test listing uploaded files."""
        magics = NativePDFMagics(mock_shell, mock_kernel)
        
        # Mock uploaded files
        mock_kernel._uploaded_files = [
            {'filename': 'test.pdf', 'type': 'pdf', 'size': 1000}
        ]
        
        # List files
        magics.llm_files_list("")
    
    def test_llm_files_clear(self, mock_kernel, mock_shell):
        """Test clearing uploaded files."""
        magics = NativePDFMagics(mock_shell, mock_kernel)
        
        # Setup
        mock_kernel.conversation_history = [
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": "123"}},
                    {"type": "text", "text": "test"}
                ]
            }
        ]
        mock_kernel._uploaded_files = [{'filename': 'test.pdf'}]
        
        # Clear
        magics.llm_files_clear("")
        
        # Verify cleared
        assert len(mock_kernel._uploaded_files) == 0


class TestCacheMagics:
    """Test cache management magic commands."""
    
    def test_llm_cache_info(self, mock_kernel, mock_shell):
        """Test showing cache info."""
        magics = CacheMagics(mock_shell, mock_kernel)
        
        # Mock cache manager
        mock_kernel.cache_manager.get_cache_info = Mock(return_value={
            'cache_dir': '/tmp/cache',
            'total_files': 5,
            'total_size': 1048576
        })
        
        # Show info
        magics.llm_cache_info("")
    
    def test_llm_cache_list(self, mock_kernel, mock_shell):
        """Test listing cached files."""
        magics = CacheMagics(mock_shell, mock_kernel)
        
        # Mock cache manager
        mock_kernel.cache_manager.list_cached_files = Mock(return_value=[
            {
                'original_name': 'test.pdf',
                'file_hash': 'abc123',
                'size': 1000,
                'cached_at': '2024-01-01'
            }
        ])
        
        # List files
        magics.llm_cache_list("")
    
    def test_llm_cache_clear(self, mock_kernel, mock_shell):
        """Test clearing cache."""
        magics = CacheMagics(mock_shell, mock_kernel)
        
        # Mock cache manager
        mock_kernel.cache_manager.clear_cache = Mock(return_value=5)
        
        # Clear with confirmation
        with patch('builtins.input', return_value='y'):
            magics.llm_cache_clear("")
        
        # Verify called
        mock_kernel.cache_manager.clear_cache.assert_called_once()


class TestMCPMagics:
    """Test MCP integration magic commands."""
    
    def test_llm_mcp_list(self, mock_kernel, mock_shell):
        """Test listing MCP servers."""
        magics = MCPMagics(mock_shell, mock_kernel)
        
        # Mock MCP manager
        mock_kernel.mcp_manager = Mock()
        mock_kernel.mcp_manager.list_servers = Mock(return_value=['server1', 'server2'])
        
        # List servers
        magics.llm_mcp_list("")
    
    def test_llm_mcp_status(self, mock_kernel, mock_shell):
        """Test showing MCP status."""
        magics = MCPMagics(mock_shell, mock_kernel)
        
        # Mock MCP manager
        mock_kernel.mcp_manager = Mock()
        mock_kernel.mcp_manager.is_connected = Mock(return_value=True)
        mock_kernel.mcp_manager.get_server_status = Mock(return_value={
            'server1': 'connected',
            'server2': 'disconnected'
        })
        
        # Show status
        magics.llm_mcp_status("")


class TestExportImport:
    """Test context export/import functionality."""
    
    def test_export_import_context(self, mock_kernel, mock_shell):
        """Test exporting and importing context."""
        magics = ContextMagics(mock_shell, mock_kernel)
        
        # Setup test data
        test_context = {
            'conversation_history': [{"role": "user", "content": "test"}],
            'context_cells': [1, 2, 3],
            'pinned_cells': [4, 5]
        }
        
        # Mock kernel methods
        mock_kernel.export_context = Mock(return_value=test_context)
        mock_kernel.import_context = Mock()
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            
        try:
            # Export
            magics.llm_export_context(temp_file)
            
            # Import
            magics.llm_import_context(temp_file)
            
            # Verify
            mock_kernel.export_context.assert_called_once()
            mock_kernel.import_context.assert_called_once()
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)


# Pytest configuration and fixtures
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment between tests."""
    yield
    # Cleanup after each test
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])