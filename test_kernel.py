"""
Pytest test suite for LLM Kernel

Run with: pytest test_kernel.py -v
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_llm_kernel(self):
        """Test LLMKernel import."""
        from llm_kernel.kernel import LLMKernel
        assert LLMKernel is not None
    
    def test_import_context_manager(self):
        """Test ContextManager and ExecutionTracker imports."""
        from llm_kernel.context_manager import ContextManager, ExecutionTracker
        assert ContextManager is not None
        assert ExecutionTracker is not None
    
    def test_import_dialogue_pruner(self):
        """Test DialoguePruner import."""
        from llm_kernel.dialogue_pruner import DialoguePruner
        assert DialoguePruner is not None
    
    def test_import_config_manager(self):
        """Test ConfigManager import."""
        from llm_kernel.config_manager import ConfigManager
        assert ConfigManager is not None


class TestConfigManager:
    """Test configuration management functionality."""
    
    def test_config_manager_creation(self):
        """Test ConfigManager can be created."""
        from llm_kernel.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        assert config_manager is not None
        assert hasattr(config_manager, 'default_config')
    
    def test_load_config(self):
        """Test configuration loading."""
        from llm_kernel.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        assert isinstance(config, dict)
        assert len(config) > 0
        assert 'default_model' in config
        assert 'context_strategy' in config
    
    def test_validate_config(self):
        """Test configuration validation."""
        from llm_kernel.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # Test valid config
        valid_config = {
            'context_strategy': 'smart',
            'max_context_tokens': 4000,
            'pruning_threshold': 0.7
        }
        validated = config_manager.validate_config(valid_config)
        assert validated['context_strategy'] == 'smart'
        assert validated['max_context_tokens'] == 4000
        
        # Test invalid config gets corrected
        invalid_config = {
            'context_strategy': 'invalid_strategy',
            'max_context_tokens': -100,
            'pruning_threshold': 2.0
        }
        validated = config_manager.validate_config(invalid_config)
        assert validated['context_strategy'] == 'smart'  # Should default
        assert validated['max_context_tokens'] == 100   # Should clamp to minimum
        assert validated['pruning_threshold'] == 1.0    # Should clamp to maximum


class TestExecutionTracker:
    """Test execution tracking functionality."""
    
    def test_execution_tracker_creation(self):
        """Test ExecutionTracker can be created."""
        from llm_kernel.context_manager import ExecutionTracker
        
        tracker = ExecutionTracker()
        assert tracker is not None
        assert len(tracker.execution_history) == 0
    
    def test_track_execution(self):
        """Test execution tracking."""
        from llm_kernel.context_manager import ExecutionTracker
        
        tracker = ExecutionTracker()
        
        # Track some executions
        tracker.track_execution("cell_1", "x = 5", 1)
        tracker.track_execution("cell_2", "y = x + 10", 2)
        
        assert len(tracker.execution_history) == 2
        assert tracker.execution_history[0].cell_id == "cell_1"
        assert tracker.execution_history[1].cell_id == "cell_2"
    
    def test_dependency_detection(self):
        """Test dependency detection."""
        from llm_kernel.context_manager import ExecutionTracker
        
        tracker = ExecutionTracker()
        
        # Track executions with dependencies
        tracker.track_execution("cell_1", "x = 5", 1)
        tracker.track_execution("cell_2", "y = x + 10", 2)
        tracker.track_execution("cell_3", "z = y * 2", 3)
        
        # Check dependencies
        deps_cell2 = tracker.get_dependencies("cell_2")
        deps_cell3 = tracker.get_dependencies("cell_3")
        
        assert "cell_1" in deps_cell2  # cell_2 uses x from cell_1
        assert "cell_2" in deps_cell3  # cell_3 uses y from cell_2
    
    def test_pin_unpin_cells(self):
        """Test cell pinning functionality."""
        from llm_kernel.context_manager import ExecutionTracker
        
        tracker = ExecutionTracker()
        
        # Test pinning
        tracker.pin_cell("cell_1")
        assert "cell_1" in tracker.pinned_cells
        
        # Test unpinning
        tracker.unpin_cell("cell_1")
        assert "cell_1" not in tracker.pinned_cells
    
    def test_ignore_include_cells(self):
        """Test cell ignoring functionality."""
        from llm_kernel.context_manager import ExecutionTracker
        
        tracker = ExecutionTracker()
        
        # Test ignoring
        tracker.ignore_cell("cell_1")
        assert "cell_1" in tracker.ignored_cells
        
        # Test including
        tracker.include_cell("cell_1")
        assert "cell_1" not in tracker.ignored_cells


class TestContextManager:
    """Test context management functionality."""
    
    def test_context_manager_creation(self):
        """Test ContextManager can be created."""
        from llm_kernel.context_manager import ContextManager
        
        context_manager = ContextManager()
        assert context_manager is not None
        assert context_manager.context_strategy == 'smart'
    
    def test_add_exchange(self):
        """Test adding exchanges to conversation history."""
        from llm_kernel.context_manager import ContextManager
        
        context_manager = ContextManager()
        
        exchange = {
            'input': 'What is Python?',
            'output': 'Python is a programming language.',
            'timestamp': 1000,
            'cell_id': 'cell_1'
        }
        
        context_manager.add_exchange(exchange)
        assert len(context_manager.conversation_history) == 1
        assert context_manager.conversation_history[0] == exchange
    
    def test_context_strategies(self):
        """Test different context strategies."""
        from llm_kernel.context_manager import ContextManager
        
        context_manager = ContextManager()
        
        # Add some sample exchanges
        exchanges = [
            {
                'input': 'What is Python?',
                'output': 'Python is a programming language.',
                'timestamp': 1000,
                'cell_id': 'cell_1'
            },
            {
                'input': 'How do I create a list?',
                'output': 'Use square brackets: my_list = [1, 2, 3]',
                'timestamp': 2000,
                'cell_id': 'cell_2'
            }
        ]
        
        for exchange in exchanges:
            context_manager.add_exchange(exchange)
        
        # Test chronological context
        context_manager.set_context_strategy('chronological')
        context = context_manager.get_chronological_context()
        assert isinstance(context, list)
        
        # Test smart context
        context_manager.set_context_strategy('smart')
        context = context_manager.get_smart_context()
        assert isinstance(context, list)
    
    def test_context_limits(self):
        """Test context window limits."""
        from llm_kernel.context_manager import ContextManager
        
        context_manager = ContextManager()
        
        # Test setting limits
        context_manager.set_context_limits(max_tokens=2000, max_cells=10)
        assert context_manager.max_context_tokens == 2000
        assert context_manager.max_context_cells == 10
    
    def test_token_estimation(self):
        """Test token estimation."""
        from llm_kernel.context_manager import ContextManager
        
        context_manager = ContextManager()
        
        exchange = {
            'input': 'What is Python?',
            'output': 'Python is a programming language.'
        }
        
        tokens = context_manager.estimate_tokens(exchange)
        assert tokens > 0
        assert isinstance(tokens, int)


class TestDialoguePruner:
    """Test dialogue pruning functionality."""
    
    def test_dialogue_pruner_creation(self):
        """Test DialoguePruner can be created."""
        from llm_kernel.dialogue_pruner import DialoguePruner
        
        pruner = DialoguePruner()
        assert pruner is not None
        assert hasattr(pruner, 'pruning_strategies')
    
    def test_pruning_strategies(self):
        """Test different pruning strategies."""
        from llm_kernel.dialogue_pruner import DialoguePruner
        
        pruner = DialoguePruner()
        
        # Create sample conversation history
        history = [
            {
                'input': 'What is Python?',
                'output': 'Python is a programming language.',
                'timestamp': 1000,
                'cell_id': 'cell_1'
            },
            {
                'input': 'How do I create a list?',
                'output': 'Use square brackets: my_list = [1, 2, 3]',
                'timestamp': 2000,
                'cell_id': 'cell_2'
            }
        ]
        
        # Test recency pruning
        pruned = pruner.prune_dialogue(history, strategy='recency')
        assert isinstance(pruned, list)
        assert len(pruned) <= len(history)
        
        # Test hybrid pruning
        pruned = pruner.prune_dialogue(history, strategy='hybrid')
        assert isinstance(pruned, list)
        assert len(pruned) <= len(history)
    
    def test_relevance_scoring(self):
        """Test relevance scoring."""
        from llm_kernel.dialogue_pruner import DialoguePruner
        
        pruner = DialoguePruner()
        
        exchange = {
            'input': 'What is Python?',
            'output': 'Python is a programming language.',
            'timestamp': 1000,
            'cell_id': 'cell_1'
        }
        
        score = pruner.calculate_multi_dimensional_score(exchange)
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_keyword_similarity(self):
        """Test keyword similarity calculation."""
        from llm_kernel.dialogue_pruner import DialoguePruner
        
        pruner = DialoguePruner()
        
        exchange = {
            'input': 'What is Python programming?',
            'output': 'Python is a programming language.'
        }
        
        query_words = {'python', 'programming', 'language'}
        similarity = pruner.calculate_keyword_similarity(exchange, query_words)
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)
        assert similarity > 0  # Should have some similarity
    
    def test_exchange_summarization(self):
        """Test exchange summarization."""
        from llm_kernel.dialogue_pruner import DialoguePruner
        
        pruner = DialoguePruner()
        
        long_exchange = {
            'input': 'This is a very long input that should be summarized. ' * 10,
            'output': 'This is a very long output that should be summarized. ' * 10,
            'timestamp': 1000,
            'cell_id': 'cell_1'
        }
        
        summary = pruner.summarize_exchange(long_exchange)
        
        assert 'is_summary' in summary
        assert summary['is_summary'] is True
        assert len(summary['input']) < len(long_exchange['input'])
        assert len(summary['output']) < len(long_exchange['output'])


class TestDependencies:
    """Test that required dependencies are available."""
    
    @pytest.mark.parametrize("package", [
        'ipykernel',
        'ipython',
        'ipywidgets',
        'dotenv',
        'requests'
    ])
    def test_required_dependencies(self, package):
        """Test that required packages can be imported."""
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package {package} is not available")
    
    @pytest.mark.parametrize("package", [
        'litellm'
    ])
    def test_optional_dependencies(self, package):
        """Test optional packages (warn if missing)."""
        try:
            __import__(package)
        except ImportError:
            pytest.skip(f"Optional package {package} is not available")


class TestKernelSpec:
    """Test kernel specification."""
    
    def test_kernel_json_exists(self):
        """Test that kernel.json exists."""
        kernel_json_path = Path(__file__).parent / "kernel.json"
        assert kernel_json_path.exists(), "kernel.json file not found"
    
    def test_kernel_json_valid(self):
        """Test that kernel.json is valid JSON."""
        kernel_json_path = Path(__file__).parent / "kernel.json"
        
        with open(kernel_json_path) as f:
            kernel_spec = json.load(f)
        
        assert isinstance(kernel_spec, dict)
        assert 'argv' in kernel_spec
        assert 'display_name' in kernel_spec
        assert 'language' in kernel_spec
    
    def test_kernel_spec_content(self):
        """Test kernel specification content."""
        kernel_json_path = Path(__file__).parent / "kernel.json"
        
        with open(kernel_json_path) as f:
            kernel_spec = json.load(f)
        
        assert kernel_spec['display_name'] == 'LLM Kernel'
        assert kernel_spec['language'] == 'python'
        assert isinstance(kernel_spec['argv'], list)
        assert len(kernel_spec['argv']) > 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    @patch('llm_kernel.kernel.litellm')
    def test_kernel_initialization(self, mock_litellm):
        """Test that the kernel can be initialized."""
        from llm_kernel.kernel import LLMKernel
        
        # Mock LiteLLM to avoid requiring API keys
        mock_litellm.completion = Mock(return_value=Mock(choices=[Mock(message=Mock(content="Test response"))]))
        
        # Create kernel instance
        kernel = LLMKernel()
        
        assert kernel is not None
        assert hasattr(kernel, 'context_manager')
        assert hasattr(kernel, 'execution_tracker')
        assert hasattr(kernel, 'dialogue_pruner')
        assert hasattr(kernel, 'config_manager')
    
    def test_end_to_end_workflow(self):
        """Test a complete workflow without external dependencies."""
        from llm_kernel.context_manager import ContextManager, ExecutionTracker
        from llm_kernel.dialogue_pruner import DialoguePruner
        
        # Create components
        context_manager = ContextManager()
        execution_tracker = ExecutionTracker()
        pruner = DialoguePruner()
        
        # Simulate workflow
        # 1. Track some executions
        execution_tracker.track_execution("cell_1", "import pandas as pd", 1)
        execution_tracker.track_execution("cell_2", "df = pd.read_csv('data.csv')", 2)
        execution_tracker.track_execution("cell_3", "result = df.describe()", 3)
        
        # 2. Add some conversation history
        exchanges = [
            {
                'input': 'Load the data',
                'output': 'I\'ll help you load the data using pandas.',
                'timestamp': 1000,
                'cell_id': 'cell_2'
            },
            {
                'input': 'Analyze the data',
                'output': 'Here\'s a statistical summary of your data.',
                'timestamp': 2000,
                'cell_id': 'cell_3'
            }
        ]
        
        for exchange in exchanges:
            context_manager.add_exchange(exchange)
        
        # 3. Test context building
        context = context_manager.get_smart_context("cell_3")
        assert isinstance(context, list)
        
        # 4. Test pruning
        pruned = pruner.prune_dialogue(context_manager.conversation_history, strategy='hybrid')
        assert isinstance(pruned, list)
        
        # 5. Test dependencies
        deps = execution_tracker.get_dependencies("cell_3")
        assert isinstance(deps, set)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
