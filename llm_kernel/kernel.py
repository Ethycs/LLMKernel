"""
Main LLM Kernel Implementation

This module contains the core LLM kernel that extends IPython kernel
with LiteLLM integration and context management capabilities.
"""

import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ipykernel.kernelbase import Kernel
from ipykernel.ipkernel import IPythonKernel
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
from IPython.display import display, HTML, JSON
import ipywidgets as widgets

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import litellm
except ImportError:
    litellm = None

from .context_manager import ContextManager, ExecutionTracker
from .dialogue_pruner import DialoguePruner
from .config_manager import ConfigManager


class LLMKernel(IPythonKernel):
    """
    Custom Jupyter kernel with LiteLLM integration and context management.
    
    Features:
    - Multi-LLM provider support via LiteLLM
    - Intelligent context window management
    - Cell dependency tracking
    - Automatic dialogue pruning
    - Magic command interface
    """
    
    implementation = 'llm_kernel'
    implementation_version = '0.1.0'
    language = 'python'
    language_version = '3.8'
    language_info = {
        'name': 'python',
        'mimetype': 'text/x-python',
        'file_extension': '.py',
    }
    banner = "LLM Kernel with Context Management - Powered by LiteLLM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize logging
        self.setup_logging()
        
        # Load environment and configuration
        self.config_manager = ConfigManager()
        self.load_environment()
        
        # Initialize core components
        self.execution_tracker = ExecutionTracker()
        self.context_manager = ContextManager()
        self.context_manager.execution_tracker = self.execution_tracker  # Link them
        self.dialogue_pruner = DialoguePruner()
        
        # LLM client management
        self.llm_clients = {}
        self.active_model = None
        self.setup_litellm()
        
        # Context and conversation state
        self.conversation_history = []
        self.model_contexts = defaultdict(list)
        
        # Async executor for parallel queries
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Register magic commands
        self.register_magic_commands()
        
        self.log.info("LLM Kernel initialized successfully")

    def setup_logging(self):
        """Configure logging for the kernel."""
        log_level = os.getenv('LLM_KERNEL_DEBUG', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format='[LLM Kernel] %(levelname)s: %(message)s'
        )
        # Ensure self.log is available even when not running in full Jupyter context
        if not hasattr(self, 'log') or self.log is None:
            self.log = logging.getLogger(__name__)

    def load_environment(self):
        """Load environment variables from .env files."""
        if load_dotenv is None:
            self.log.warning("python-dotenv not installed. Using system environment only.")
            return
            
        # Look for .env files in current directory and parent directories
        env_path = self.find_env_file()
        if env_path:
            load_dotenv(env_path)
            self.log.info(f"Loaded environment from {env_path}")
        else:
            self.log.warning("No .env file found, using system environment")

    def find_env_file(self) -> Optional[Path]:
        """Find .env file in current directory or parent directories."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            env_file = parent / '.env'
            if env_file.exists():
                return env_file
        return None

    def setup_litellm(self):
        """Initialize LiteLLM clients for different providers."""
        if litellm is None:
            self.log.error("LiteLLM not installed. Please install with: pip install litellm")
            return
            
        # Configure available models based on API keys
        available_models = {}
        
        if os.getenv('OPENAI_API_KEY'):
            available_models.update({
                'gpt-4o': 'gpt-4o',
                'gpt-4o-mini': 'gpt-4o-mini',
                'gpt-4': 'gpt-4',
                'gpt-3.5-turbo': 'gpt-3.5-turbo'
            })
            
        if os.getenv('ANTHROPIC_API_KEY'):
            available_models.update({
                'claude-3-opus': 'claude-3-opus-20240229',
                'claude-3-sonnet': 'claude-3-sonnet-20240229',
                'claude-3-haiku': 'claude-3-haiku-20240307'
            })
            
        if os.getenv('GOOGLE_API_KEY'):
            available_models.update({
                'gemini-pro': 'gemini-pro',
                'gemini-1.5-pro': 'gemini-1.5-pro'
            })
            
        # Add local models if Ollama is available
        ollama_base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        if self.check_ollama_available(ollama_base):
            available_models.update({
                'llama3': 'ollama/llama3',
                'codellama': 'ollama/codellama'
            })
        
        self.llm_clients = available_models
        
        # Set default model
        default_model = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o-mini')
        if default_model in self.llm_clients:
            self.active_model = default_model
        elif self.llm_clients:
            self.active_model = list(self.llm_clients.keys())[0]
        else:
            self.log.error("No API keys found. Please configure your .env file.")
            
        self.log.info(f"Available models: {list(self.llm_clients.keys())}")
        self.log.info(f"Active model: {self.active_model}")

    def check_ollama_available(self, base_url: str) -> bool:
        """Check if Ollama is available at the given URL."""
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def register_magic_commands(self):
        """Register magic commands with the kernel."""
        magic_instance = LLMKernelMagics(self.shell, self)
        self.shell.register_magics(magic_instance)

    async def query_llm_async(self, query: str, model: str = None, **kwargs) -> str:
        """Asynchronously query an LLM model."""
        if model is None:
            model = self.active_model
            
        if model not in self.llm_clients:
            raise ValueError(f"Model {model} not available. Available: {list(self.llm_clients.keys())}")
            
        model_name = self.llm_clients[model]
        
        # Get context for this model
        context = self.context_manager.get_context_for_model(model)
        
        # Build messages
        messages = []
        for ctx in context:
            messages.append({"role": "user", "content": ctx['input']})
            if ctx.get('output'):
                messages.append({"role": "assistant", "content": ctx['output']})
        
        messages.append({"role": "user", "content": query})
        
        try:
            # Use LiteLLM to query the model
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: litellm.completion(
                    model=model_name,
                    messages=messages,
                    **kwargs
                )
            )
            
            result = response.choices[0].message.content
            
            # Track the exchange
            self.track_exchange(model, query, result)
            
            return result
            
        except Exception as e:
            self.log.error(f"Error querying {model}: {e}")
            return f"Error: {str(e)}"

    def track_exchange(self, model: str, query: str, response: str):
        """Track an LLM exchange for context management."""
        exchange = {
            'model': model,
            'input': query,
            'output': response,
            'timestamp': time.time(),
            'cell_id': getattr(self, '_current_cell_id', None)
        }
        
        self.conversation_history.append(exchange)
        self.model_contexts[model].append(exchange)
        
        # Update context manager
        self.context_manager.add_exchange(exchange)

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        """Override execute to track cell execution."""
        # Track cell execution
        cell_id = f"cell_{len(self.execution_tracker.execution_history)}"
        self._current_cell_id = cell_id
        
        self.execution_tracker.track_execution(cell_id, code, self.execution_count)
        
        # Call parent execute
        return super().do_execute(code, silent, store_history, user_expressions, allow_stdin)


@magics_class
class LLMKernelMagics(Magics):
    """Magic commands for the LLM kernel."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance

    @line_magic
    def llm_models(self, line):
        """List available LLM models."""
        if not self.kernel.llm_clients:
            print("❌ No models available. Please configure API keys in .env file.")
            return
            
        print("🤖 Available LLM Models:")
        for model in self.kernel.llm_clients.keys():
            status = "✅ (active)" if model == self.kernel.active_model else "⚪"
            print(f"  {status} {model}")

    @line_magic
    def llm_model(self, line):
        """Switch active model: %llm_model gpt-4o"""
        model = line.strip()
        if not model:
            print(f"Current model: {self.kernel.active_model}")
            return
            
        if model in self.kernel.llm_clients:
            self.kernel.active_model = model
            print(f"✅ Switched to {model}")
        else:
            print(f"❌ Unknown model '{model}'. Available: {list(self.kernel.llm_clients.keys())}")

    @cell_magic
    def llm(self, line, cell):
        """Query the active LLM model."""
        # Parse arguments
        args = line.split() if line else []
        model = None
        
        for arg in args:
            if arg.startswith('--model='):
                model = arg.split('=', 1)[1]
        
        # Query the model
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(
            self.kernel.query_llm_async(cell, model)
        )
        
        print(result)
        return result

    @cell_magic
    def llm_gpt4(self, line, cell):
        """Query GPT-4 specifically."""
        return self.llm('--model=gpt-4o', cell)

    @cell_magic
    def llm_claude(self, line, cell):
        """Query Claude specifically."""
        return self.llm('--model=claude-3-sonnet', cell)

    @cell_magic
    def llm_compare(self, line, cell):
        """Compare responses from multiple models."""
        models = line.split() if line else ['gpt-4o-mini', 'claude-3-haiku']
        
        # Filter to available models
        available_models = [m for m in models if m in self.kernel.llm_clients]
        
        if not available_models:
            print("❌ No available models specified")
            return
            
        print(f"🔄 Querying {len(available_models)} models...")
        
        # Query models in parallel
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        tasks = [
            self.kernel.query_llm_async(cell, model)
            for model in available_models
        ]
        
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        # Display results
        self.display_comparison(dict(zip(available_models, results)))

    def display_comparison(self, results):
        """Display side-by-side model comparison."""
        # Create tabs for each model
        tab_children = []
        tab_titles = []
        
        for model, response in results.items():
            content = widgets.HTML(f"""
            <div style="padding: 15px;">
                <h4>{model.upper()}</h4>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6;">
                    <pre style="white-space: pre-wrap; margin: 0;">{response}</pre>
                </div>
            </div>
            """)
            tab_children.append(content)
            tab_titles.append(model)
        
        tabs = widgets.Tab(children=tab_children)
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
        
        display(tabs)

    @line_magic
    def llm_status(self, line):
        """Show current context and model status."""
        print("🤖 LLM Kernel Status")
        print("=" * 50)
        print(f"Active Model: {self.kernel.active_model}")
        print(f"Available Models: {len(self.kernel.llm_clients)}")
        print(f"Conversation History: {len(self.kernel.conversation_history)} exchanges")
        
        if self.kernel.conversation_history:
            total_tokens = sum(
                len(ex['input'].split()) + len(ex.get('output', '').split())
                for ex in self.kernel.conversation_history
            )
            print(f"Estimated Tokens: ~{total_tokens}")

    @line_magic
    def llm_clear(self, line):
        """Clear conversation history."""
        self.kernel.conversation_history.clear()
        self.kernel.model_contexts.clear()
        print("✅ Conversation history cleared")

    @line_magic
    def llm_pin_cell(self, line):
        """Pin a cell for context: %llm_pin_cell 5"""
        cell_id = line.strip()
        if not cell_id:
            print("❌ Please specify a cell ID")
            return
        
        self.kernel.execution_tracker.pin_cell(cell_id)
        print(f"📌 Pinned cell {cell_id}")

    @line_magic
    def llm_unpin_cell(self, line):
        """Unpin a cell: %llm_unpin_cell 5"""
        cell_id = line.strip()
        if not cell_id:
            print("❌ Please specify a cell ID")
            return
        
        self.kernel.execution_tracker.unpin_cell(cell_id)
        print(f"📌 Unpinned cell {cell_id}")

    @line_magic
    def llm_context(self, line):
        """Set context strategy: %llm_context smart"""
        strategy = line.strip()
        if not strategy:
            print(f"Current strategy: {self.kernel.context_manager.context_strategy}")
            print("Available: chronological, dependency, smart, manual")
            return
        
        try:
            self.kernel.context_manager.set_context_strategy(strategy)
            print(f"✅ Context strategy set to: {strategy}")
        except ValueError as e:
            print(f"❌ {e}")

    @line_magic
    def llm_prune(self, line):
        """Prune conversation history: %llm_prune --strategy=hybrid"""
        # Parse arguments
        args = line.split() if line else []
        strategy = 'hybrid'
        threshold = 0.7
        
        for arg in args:
            if arg.startswith('--strategy='):
                strategy = arg.split('=', 1)[1]
            elif arg.startswith('--threshold='):
                threshold = float(arg.split('=', 1)[1])
        
        # Perform pruning
        original_count = len(self.kernel.conversation_history)
        pruned_history = self.kernel.dialogue_pruner.prune_dialogue(
            self.kernel.conversation_history,
            strategy=strategy,
            threshold=threshold
        )
        
        # Update history
        self.kernel.conversation_history = pruned_history
        pruned_count = original_count - len(pruned_history)
        
        print(f"✂️ Pruned {pruned_count} exchanges using {strategy} strategy")
        print(f"📊 Context size: {original_count} → {len(pruned_history)} exchanges")

    @line_magic
    def llm_config(self, line):
        """Show interactive configuration panel."""
        self.create_config_panel()

    def create_config_panel(self):
        """Create interactive configuration panel."""
        # Model selection
        model_selector = widgets.Dropdown(
            options=list(self.kernel.llm_clients.keys()),
            value=self.kernel.active_model,
            description='Active Model:'
        )
        
        # Status display
        status_html = widgets.HTML(
            value=f"""
            <div style="padding: 10px; background: #f8f9fa; border-radius: 4px;">
                <strong>Status:</strong><br>
                Models: {len(self.kernel.llm_clients)}<br>
                History: {len(self.kernel.conversation_history)} exchanges
            </div>
            """
        )
        
        # Action buttons
        clear_btn = widgets.Button(
            description='🗑️ Clear History',
            button_style='warning'
        )
        
        def on_model_change(change):
            self.kernel.active_model = change['new']
            print(f"✅ Switched to {change['new']}")
        
        def on_clear_click(b):
            self.kernel.conversation_history.clear()
            self.kernel.model_contexts.clear()
            print("✅ History cleared")
        
        model_selector.observe(on_model_change, names='value')
        clear_btn.on_click(on_clear_click)
        
        # Layout
        config_panel = widgets.VBox([
            widgets.HTML('<h3>🤖 LLM Kernel Configuration</h3>'),
            model_selector,
            status_html,
            clear_btn
        ])
        
        display(config_panel)
