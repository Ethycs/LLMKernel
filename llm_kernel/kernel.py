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
from .mcp_manager import MCPManager


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
        
        # Enable VS Code debugging if requested
        self.setup_debugging()
        
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
        
        # Display options
        self.display_mode = 'inline'  # 'inline' or 'chat'
        
        # Context persistence
        self.context_persistence = True  # Load previous cells by default
        self.saved_context = None  # For manual context saves
        
        # Async executor for parallel queries
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # MCP (Model Context Protocol) manager
        self.mcp_manager = MCPManager(logger=self.log)
        self.mcp_manager.load_config()
        
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
    
    def setup_debugging(self):
        """Setup VS Code debugging if enabled."""
        if os.getenv('LLM_KERNEL_DEBUGGER') == 'true':
            try:
                import debugpy
                port = int(os.getenv('LLM_KERNEL_DEBUG_PORT', '5678'))
                
                # Use 0.0.0.0 to allow connections from any interface
                debugpy.listen(("0.0.0.0", port))
                self.log.info(f"Debugger listening on 0.0.0.0:{port}")
                
                # Wait for debugger if requested
                if os.getenv('LLM_KERNEL_DEBUG_WAIT') == 'true':
                    self.log.info("Waiting for debugger to attach...")
                    debugpy.wait_for_client()
                    self.log.info("Debugger attached!")
            except ImportError:
                self.log.warning("debugpy not installed - install with: pixi install -e notebook")
            except Exception as e:
                self.log.warning(f"Could not start debugger: {e}")

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
    
    def get_notebook_path(self):
        """Try to get the current notebook path."""
        try:
            # Try to get from IPython
            if hasattr(self.shell, 'user_ns') and 'notebook_path' in self.shell.user_ns:
                return self.shell.user_ns['notebook_path']
            
            # Try to get from kernel connection file
            connection_file = self.config.get('IPKernelApp', {}).get('connection_file', '')
            if connection_file:
                # Extract notebook path from connection info if available
                import re
                match = re.search(r'kernel-(.+)\.json', connection_file)
                if match:
                    # This is a heuristic - may need adjustment
                    pass
                    
        except Exception as e:
            self.log.debug(f"Could not determine notebook path: {e}")
        
        return None
    
    def load_notebook_cells(self, notebook_path):
        """Load cells from a notebook file."""
        try:
            import json
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            cells = []
            for cell in notebook.get('cells', []):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                    outputs = []
                    
                    # Extract text outputs
                    for output in cell.get('outputs', []):
                        if output.get('output_type') == 'stream':
                            text = ''.join(output.get('text', []))
                            outputs.append(text)
                        elif output.get('output_type') == 'execute_result':
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                outputs.append(data['text/plain'])
                    
                    cells.append({
                        'type': 'code',
                        'source': source,
                        'output': '\n'.join(outputs) if outputs else None
                    })
                elif cell['cell_type'] == 'markdown':
                    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                    cells.append({
                        'type': 'markdown',
                        'source': source,
                        'output': None
                    })
            
            return cells
        except Exception as e:
            self.log.warning(f"Could not load notebook cells: {e}")
            return []
    
    def get_notebook_cells_as_context(self, max_cells=50, include_all_cells=True):
        """Get notebook cells as context for LLM.
        
        Returns a list of messages built from notebook cells where:
        - Code cells with no output are user messages
        - Code cells with output become user/assistant pairs
        - Markdown cells are system messages for context
        
        Args:
            max_cells: Maximum number of cells to include
            include_all_cells: If True, includes cells from previous sessions too
        """
        # Check if we have reranked context
        if hasattr(self, '_reranked_context') and self._reranked_context:
            return self._reranked_context
            
        messages = []
        
        # Try to get notebook cells from the shell's history
        try:
            if hasattr(self.shell, 'history_manager'):
                if include_all_cells:
                    # Get ALL cells from the notebook, not just current session
                    # This includes cells from previous kernel sessions
                    hist = list(self.shell.history_manager.get_range(
                        session=0,  # 0 means all sessions
                        start=1,
                        stop=None,
                        raw=True,
                        output=True
                    ))
                else:
                    # Get only current session cells
                    hist = list(self.shell.history_manager.get_range(
                        session=self.shell.history_manager.session_number,
                        start=1,
                        stop=None,
                        raw=True,
                        output=True
                    ))
                
                # Process history into messages
                cell_count = 0
                for session, line_num, (input_text, output) in hist[-max_cells:]:
                    if not input_text.strip():
                        continue
                    
                    # Check if this cell is hidden
                    cell_id = f"cell_{cell_count}"
                    if hasattr(self, 'hidden_cells') and cell_id in self.hidden_cells:
                        cell_count += 1
                        continue
                        
                    # Skip pure magic commands (but include %%llm queries)
                    if input_text.strip().startswith('%') and not input_text.strip().startswith('%%llm'):
                        cell_count += 1
                        continue
                    
                    # Skip %%hide cells
                    if input_text.strip().startswith('%%hide'):
                        cell_count += 1
                        continue
                    
                    # For %%llm queries, extract just the content after the magic
                    if input_text.strip().startswith('%%llm'):
                        lines = input_text.strip().split('\n', 1)
                        if len(lines) > 1:
                            input_text = lines[1]  # Get content after %%llm line
                        else:
                            cell_count += 1
                            continue
                    
                    # Add user message
                    messages.append({"role": "user", "content": input_text.strip()})
                    
                    # Add assistant message if there's output
                    # Filter out status messages and only include actual responses
                    if output and not str(output).startswith('[') and not str(output).startswith('💬'):
                        # Clean up output - remove status lines
                        output_lines = str(output).split('\n')
                        cleaned_lines = []
                        for line in output_lines:
                            # Skip status indicators
                            if (line.startswith('---') or 
                                line.startswith('===') or
                                line.startswith('💬') or
                                line.startswith('🙈') or
                                line.startswith('[') and line.endswith(']')):
                                continue
                            cleaned_lines.append(line)
                        
                        cleaned_output = '\n'.join(cleaned_lines).strip()
                        if cleaned_output:
                            messages.append({"role": "assistant", "content": cleaned_output})
                    
                    cell_count += 1
                        
        except Exception as e:
            self.log.warning(f"Could not get notebook history: {e}")
            
        return messages

    async def query_llm_async(self, query: str, model: str = None, **kwargs) -> str:
        """Asynchronously query an LLM model."""
        if model is None:
            model = self.active_model
            
        if model not in self.llm_clients:
            raise ValueError(f"Model {model} not available. Available: {list(self.llm_clients.keys())}")
            
        model_name = self.llm_clients[model]
        
        # Check if we're in notebook context mode
        use_notebook_context = getattr(self, 'notebook_context_mode', False)
        
        if use_notebook_context:
            # Use notebook cells as context
            messages = self.get_notebook_cells_as_context()
            messages.append({"role": "user", "content": query})
        else:
            # Use traditional context manager
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
            
            # Track costs if available
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                if not hasattr(self, 'session_costs'):
                    self.session_costs = {'total': 0.0, 'by_model': {}}
                
                cost = response._hidden_params['response_cost']
                self.session_costs['total'] += cost
                if model not in self.session_costs['by_model']:
                    self.session_costs['by_model'][model] = 0.0
                self.session_costs['by_model'][model] += cost
            
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
        
        # Check if we're in chat mode and this isn't a magic command
        if (hasattr(self, 'chat_mode') and self.chat_mode and 
            not code.strip().startswith('%') and 
            not code.strip().startswith('!') and
            code.strip() and
            not code.strip().startswith('#')):
            
            # In chat mode, treat non-magic, non-empty cells as LLM queries
            try:
                # Find the LLMKernelMagics instance
                magic_instance = None
                for mc in self.shell.magics_manager.registry.values():
                    if isinstance(mc, LLMKernelMagics):
                        magic_instance = mc
                        break
                
                if magic_instance:
                    # Call the llm method directly
                    magic_instance.llm('', code)
                    # Return successful execution
                    return {'status': 'ok', 'execution_count': self.execution_count,
                            'payload': [], 'user_expressions': {}}
                else:
                    self.log.error("Could not find LLMKernelMagics instance")
            except Exception as e:
                self.log.error(f"Error in chat mode: {e}", exc_info=True)
        
        # Normal execution for magic commands, empty cells, or when not in chat mode
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
        # Handle async properly in Jupyter environments
        import nest_asyncio
        nest_asyncio.apply()
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in Jupyter, use the existing loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.kernel.query_llm_async(cell, model))
                    result = future.result()
            else:
                # Normal execution
                result = loop.run_until_complete(
                    self.kernel.query_llm_async(cell, model)
                )
        except RuntimeError:
            # Fallback: create new event loop
            result = asyncio.run(self.kernel.query_llm_async(cell, model))
        except Exception as e:
            print(f"❌ Error querying LLM: {e}")
            return None
        
        # Display based on mode
        if self.kernel.display_mode == 'chat':
            # Chat mode - clean cell-based display
            model_name = model or self.kernel.active_model
            
            # Show model info
            print(f"[{model_name}]")
            print("-" * 40)
            
            # Show the response
            print(result)
            
            # # Add spacing and continuation hint
            # print("\n" + "=" * 40)
            # print("💬 Continue in next cell with %%llm")
            
            # Track this as part of a conversation thread
            if not hasattr(self.kernel, '_chat_thread_id'):
                import uuid
                self.kernel._chat_thread_id = str(uuid.uuid4())[:8]
            
            # Store thread metadata
            if hasattr(self.kernel, '_current_cell_id'):
                cell_id = self.kernel._current_cell_id
                if not hasattr(self.kernel.execution_tracker, 'chat_threads'):
                    self.kernel.execution_tracker.chat_threads = {}
                self.kernel.execution_tracker.chat_threads[cell_id] = self.kernel._chat_thread_id
        else:
            # Default inline display
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
    def hide(self, line, cell):
        """Hide this cell from the LLM context window.
        
        Usage:
            %%hide
            # This content won't be included in the context
            secret_api_key = "..."
        """
        # Execute the cell normally but mark it as hidden
        if hasattr(self.kernel, '_current_cell_id'):
            cell_id = self.kernel._current_cell_id
            
            # Initialize hidden cells set if not exists
            if not hasattr(self.kernel, 'hidden_cells'):
                self.kernel.hidden_cells = set()
            
            # Add this cell to hidden cells
            self.kernel.hidden_cells.add(cell_id)
            
            # Also store in execution tracker for persistence
            if hasattr(self.kernel.execution_tracker, 'hidden_cells'):
                self.kernel.execution_tracker.hidden_cells.add(cell_id)
            else:
                self.kernel.execution_tracker.hidden_cells = {cell_id}
        
        # Execute the cell content normally
        self.shell.run_cell(cell)
        
        # Show indicator that cell is hidden
        print("🙈 Cell hidden from LLM context")
    
    @cell_magic
    def meta(self, line, cell):
        """Define custom context processing functions.
        
        Usage:
            %%meta ranking
            # Define a custom ranking function
            def rank_cells(messages, query):
                # Your custom logic here
                return reordered_messages
                
            %%meta filter
            # Define a custom filter function
            def filter_cells(messages):
                # Your custom logic here
                return filtered_messages
                
            %%meta transform
            # Define a custom transform function
            def transform_context(messages):
                # Your custom logic here
                return transformed_messages
        """
        meta_type = line.strip().lower() if line else 'ranking'
        
        # Execute the cell to define the function
        self.shell.run_cell(cell)
        
        # Store the meta function based on type
        if not hasattr(self.kernel, '_meta_functions'):
            self.kernel._meta_functions = {}
        
        # Try to extract the defined function from the cell
        import ast
        try:
            tree = ast.parse(cell)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    # Get the function from the namespace
                    if func_name in self.shell.user_ns:
                        func = self.shell.user_ns[func_name]
                        self.kernel._meta_functions[meta_type] = func
                        print(f"✅ Registered {meta_type} function: {func_name}")
                        
                        # Show expected signature
                        if meta_type == 'ranking':
                            print("   Expected signature: rank_cells(messages, query) -> reordered_messages")
                        elif meta_type == 'filter':
                            print("   Expected signature: filter_cells(messages) -> filtered_messages")
                        elif meta_type == 'transform':
                            print("   Expected signature: transform_context(messages) -> transformed_messages")
                        return
                        
        except Exception as e:
            print(f"❌ Error parsing meta function: {e}")
            
        print(f"⚠️  No function found in cell. Define a function for {meta_type}")
    
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
    def llm_history(self, line):
        """Show conversation history as a static chat window."""
        args = line.split() if line else []
        show_cells = '--cells' in args
        show_all = '--all' in args
        
        if not self.kernel.conversation_history:
            print("💬 No conversation history yet")
            return
        
        # Build HTML chat display
        html_parts = ['<div style="font-family: Arial, sans-serif; max-width: 800px;">']
        html_parts.append('<h3>💬 LLM Conversation History</h3>')
        
        # Sort by notebook order if available
        if hasattr(self.kernel.context_manager, 'sort_by_notebook_order'):
            sorted_history = self.kernel.context_manager.sort_by_notebook_order(
                self.kernel.conversation_history.copy()
            )
        else:
            # Fallback to chronological order
            sorted_history = self.kernel.conversation_history.copy()
        
        for i, exchange in enumerate(sorted_history):
            cell_info = ""
            if show_cells and exchange.get('cell_id'):
                cell_num = self.kernel.execution_tracker.cell_id_to_number.get(
                    exchange['cell_id'], '?'
                )
                cell_info = f" <small style='color: #666;'>[Cell {cell_num}]</small>"
            
            # User message
            import html as html_module
            escaped_input = html_module.escape(exchange['input'])
            html_parts.append(f'''
            <div style="margin: 10px 0; padding: 10px; background: #e3f2fd; border-radius: 10px;">
                <strong>👤 User{cell_info}:</strong><br>
                <pre style="white-space: pre-wrap; margin: 5px 0;">{escaped_input}</pre>
            </div>
            ''')
            
            # Assistant response
            if exchange.get('output'):
                model = exchange.get('model', 'Unknown')
                escaped_output = html_module.escape(exchange['output'])
                html_parts.append(f'''
                <div style="margin: 10px 0 20px 40px; padding: 10px; background: #f5f5f5; border-radius: 10px;">
                    <strong>🤖 {model}:</strong><br>
                    <pre style="white-space: pre-wrap; margin: 5px 0;">{escaped_output}</pre>
                </div>
                ''')
        
        html_parts.append('</div>')
        display(HTML(''.join(html_parts)))

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
    def llm_debug(self, line):
        """Enable debugging: %llm_debug [port]"""
        try:
            import debugpy
            port = int(line.strip()) if line.strip() else 5678
            
            # Check if already debugging
            if debugpy.is_client_connected():
                print("✅ Debugger already connected")
                return
                
            # Start debug server
            try:
                debugpy.listen(("0.0.0.0", port))
                print(f"🐛 Debugger listening on port {port}")
                print("📎 In VS Code: Run and Debug → 'Debug LLM Kernel' → Start Debugging (F5)")
                
                # Optional: wait for attach
                if '--wait' in line:
                    print("⏸️  Waiting for debugger to attach...")
                    debugpy.wait_for_client()
                    print("✅ Debugger attached!")
            except RuntimeError as e:
                if "already listening" in str(e):
                    print(f"ℹ️  Debugger already listening on port {port}")
                else:
                    raise
                    
        except ImportError:
            print("❌ debugpy not installed")
            print("Run: pixi install -e notebook")
        except Exception as e:
            print(f"❌ Error starting debugger: {e}")
    
    @line_magic
    def llm_chat(self, line):
        """Toggle chat mode on/off or check status"""
        arg = line.strip().lower()
        
        if not arg:
            # Toggle mode
            if hasattr(self.kernel, 'chat_mode') and self.kernel.chat_mode:
                self.kernel.chat_mode = False
                self.kernel.display_mode = 'inline'
                self.kernel.notebook_context_mode = False
                print("💬 Chat mode: OFF")
                print("📓 Notebook context mode: OFF")
            else:
                self.kernel.chat_mode = True
                self.kernel.display_mode = 'chat'
                self.kernel.notebook_context_mode = True  # Enable notebook context
                print("💬 Chat mode: ON")
                print("📓 Notebook context mode: ON")
                print("📝 Just type in any cell to chat!")
                print("💡 Your notebook cells are now the LLM's context window!")
        elif arg in ['on', 'true', '1']:
            self.kernel.chat_mode = True
            self.kernel.display_mode = 'chat'
            self.kernel.notebook_context_mode = True
            print("💬 Chat mode: ON")
            print("📓 Notebook context mode: ON")
            print("📝 Just type in any cell to chat!")
            print("💡 Your notebook cells are now the LLM's context window!")
        elif arg in ['off', 'false', '0']:
            self.kernel.chat_mode = False
            self.kernel.display_mode = 'inline'
            self.kernel.notebook_context_mode = False
            print("💬 Chat mode: OFF")
            print("📓 Notebook context mode: OFF")
        elif arg == 'status':
            status = "ON" if hasattr(self.kernel, 'chat_mode') and self.kernel.chat_mode else "OFF"
            print(f"💬 Chat mode: {status}")
        else:
            print("Usage: %llm_chat [on|off|status]")
            print("       %llm_chat  (toggles mode)")
    
    @line_magic
    def llm_display(self, line):
        """Set display mode: %llm_display chat"""
        mode = line.strip()
        if not mode:
            print(f"Current display mode: {self.kernel.display_mode}")
            print("Available modes: inline, chat")
            return
        
        if mode in ['inline', 'chat']:
            self.kernel.display_mode = mode
            print(f"✅ Display mode set to: {mode}")
            
            if mode == 'chat':
                # Also enable chat mode
                self.kernel.chat_mode = True
                print("💬 Chat mode enabled - use %%llm to continue conversations")
        else:
            print(f"❌ Invalid mode. Use 'inline' or 'chat'")
    
    @line_magic
    def llm_notebook_context(self, line):
        """Toggle notebook context mode on/off
        
        When enabled, the LLM sees notebook cells as its context window
        instead of using the traditional conversation history.
        """
        arg = line.strip().lower()
        
        if not arg:
            # Toggle mode
            current = getattr(self.kernel, 'notebook_context_mode', False)
            self.kernel.notebook_context_mode = not current
            status = "ON" if self.kernel.notebook_context_mode else "OFF"
            print(f"📓 Notebook context mode: {status}")
            if self.kernel.notebook_context_mode:
                print("📝 The LLM now sees your notebook cells as context!")
                print("💡 Each cell becomes part of the conversation")
        elif arg in ['on', 'true', '1']:
            self.kernel.notebook_context_mode = True
            print("📓 Notebook context mode: ON")
            print("📝 The LLM now sees your notebook cells as context!")
        elif arg in ['off', 'false', '0']:
            self.kernel.notebook_context_mode = False
            print("📓 Notebook context mode: OFF")
            print("📝 Using traditional conversation history")
        elif arg == 'status':
            status = "ON" if getattr(self.kernel, 'notebook_context_mode', False) else "OFF"
            print(f"📓 Notebook context mode: {status}")
        else:
            print("Usage: %llm_notebook_context [on|off|status]")
            print("       %llm_notebook_context  (toggles mode)")

    @line_magic
    def llm_unhide(self, line):
        """Unhide a cell from the LLM context.
        
        Usage:
            %llm_unhide 5  # Unhide cell 5
            %llm_unhide all  # Unhide all cells
        """
        arg = line.strip()
        
        if not hasattr(self.kernel, 'hidden_cells'):
            self.kernel.hidden_cells = set()
            
        if arg == 'all':
            count = len(self.kernel.hidden_cells)
            self.kernel.hidden_cells.clear()
            if hasattr(self.kernel.execution_tracker, 'hidden_cells'):
                self.kernel.execution_tracker.hidden_cells.clear()
            print(f"👁️  Unhidden {count} cells")
        elif arg.isdigit():
            cell_id = f"cell_{arg}"
            if cell_id in self.kernel.hidden_cells:
                self.kernel.hidden_cells.remove(cell_id)
                if hasattr(self.kernel.execution_tracker, 'hidden_cells'):
                    self.kernel.execution_tracker.hidden_cells.discard(cell_id)
                print(f"👁️  Unhidden cell {arg}")
            else:
                print(f"ℹ️  Cell {arg} was not hidden")
        else:
            print("Usage: %llm_unhide <cell_number> or %llm_unhide all")
    
    @line_magic
    def llm_hidden(self, line):
        """Show which cells are currently hidden from context."""
        if not hasattr(self.kernel, 'hidden_cells') or not self.kernel.hidden_cells:
            print("No cells are currently hidden")
        else:
            hidden_nums = sorted([int(cell_id.split('_')[1]) for cell_id in self.kernel.hidden_cells])
            print(f"🙈 Hidden cells: {', '.join(map(str, hidden_nums))}")

    @line_magic
    def llm_context(self, line):
        """Show current context that will be sent to the LLM"""
        if getattr(self.kernel, 'notebook_context_mode', False):
            print("📓 Notebook Context Mode - Showing cells that will be sent to LLM:")
            print("=" * 60)
            
            messages = self.kernel.get_notebook_cells_as_context()
            
            if not messages:
                print("No context available yet. Start typing in cells!")
            else:
                for i, msg in enumerate(messages):
                    role = msg['role'].upper()
                    content = msg['content']
                    
                    # Truncate long content for display
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    print(f"\n[{i+1}] {role}:")
                    print(content)
                    print("-" * 40)
                    
                print(f"\nTotal messages: {len(messages)}")
                
                # Get accurate token count using litellm
                if litellm and self.kernel.active_model:
                    try:
                        token_count = litellm.token_counter(
                            model=self.kernel.active_model,
                            messages=messages
                        )
                        max_tokens = litellm.get_max_tokens(self.kernel.active_model)
                        usage_pct = (token_count / max_tokens * 100) if max_tokens > 0 else 0
                        
                        print(f"Token usage: {token_count:,} / {max_tokens:,} ({usage_pct:.1f}%)")
                        
                        # Visual progress bar
                        bar_width = 30
                        filled = int(bar_width * usage_pct / 100)
                        bar = "█" * filled + "░" * (bar_width - filled)
                        print(f"[{bar}]")
                        
                        if usage_pct > 80:
                            print("⚠️  Warning: Context window nearly full!")
                    except:
                        # Fallback to estimate
                        total_chars = sum(len(msg['content']) for msg in messages)
                        estimated_tokens = total_chars // 4
                        print(f"Estimated tokens: ~{estimated_tokens}")
                else:
                    # Fallback to estimate
                    total_chars = sum(len(msg['content']) for msg in messages)
                    estimated_tokens = total_chars // 4
                    print(f"Estimated tokens: ~{estimated_tokens}")
                
                # Show hidden cells if any
                if hasattr(self.kernel, 'hidden_cells') and self.kernel.hidden_cells:
                    hidden_nums = sorted([int(cell_id.split('_')[1]) for cell_id in self.kernel.hidden_cells])
                    print(f"\n🙈 Hidden cells: {', '.join(map(str, hidden_nums))}")
        else:
            print("📝 Traditional Context Mode")
            print("Use %llm_status to see conversation history")
    
    @line_magic
    def llm_context_save(self, line):
        """Save current context to a file.
        
        Usage:
            %llm_context_save           # Save to auto-named file
            %llm_context_save my_context.json  # Save to specific file
        """
        filename = line.strip() or f"llm_context_{int(time.time())}.json"
        
        try:
            import json
            context_data = {
                'messages': self.kernel.get_notebook_cells_as_context(),
                'hidden_cells': list(getattr(self.kernel, 'hidden_cells', set())),
                'conversation_history': self.kernel.conversation_history,
                'metadata': {
                    'saved_at': time.time(),
                    'kernel_version': self.kernel.implementation_version,
                    'active_model': self.kernel.active_model,
                    'notebook_context_mode': getattr(self.kernel, 'notebook_context_mode', False)
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2)
            
            print(f"💾 Context saved to: {filename}")
            print(f"   Messages: {len(context_data['messages'])}")
            print(f"   Hidden cells: {len(context_data['hidden_cells'])}")
            
        except Exception as e:
            print(f"❌ Error saving context: {e}")
    
    @line_magic
    def llm_context_load(self, line):
        """Load context from a file.
        
        Usage:
            %llm_context_load my_context.json
        """
        filename = line.strip()
        if not filename:
            print("Usage: %llm_context_load <filename>")
            return
            
        try:
            import json
            with open(filename, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
            
            # Restore context
            if 'conversation_history' in context_data:
                self.kernel.conversation_history = context_data['conversation_history']
            
            if 'hidden_cells' in context_data:
                self.kernel.hidden_cells = set(context_data['hidden_cells'])
            
            # Store loaded context for reference
            self.kernel.saved_context = context_data.get('messages', [])
            
            print(f"📂 Context loaded from: {filename}")
            print(f"   Messages: {len(context_data.get('messages', []))}")
            print(f"   Hidden cells: {len(context_data.get('hidden_cells', []))}")
            
            metadata = context_data.get('metadata', {})
            if metadata:
                saved_time = metadata.get('saved_at', 0)
                if saved_time:
                    from datetime import datetime
                    saved_date = datetime.fromtimestamp(saved_time).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"   Saved at: {saved_date}")
                    
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
        except Exception as e:
            print(f"❌ Error loading context: {e}")
    
    @line_magic
    def llm_context_reset(self, line):
        """Reset context to a clean state.
        
        Usage:
            %llm_context_reset         # Clear all context
            %llm_context_reset --keep-hidden  # Clear context but keep hidden cells
        """
        keep_hidden = '--keep-hidden' in line
        
        # Clear conversation history
        self.kernel.conversation_history.clear()
        self.kernel.model_contexts.clear()
        
        # Clear saved context
        self.kernel.saved_context = None
        
        # Optionally clear hidden cells
        if not keep_hidden and hasattr(self.kernel, 'hidden_cells'):
            self.kernel.hidden_cells.clear()
            
        print("🔄 Context reset")
        if keep_hidden and hasattr(self.kernel, 'hidden_cells') and self.kernel.hidden_cells:
            print(f"   Kept {len(self.kernel.hidden_cells)} hidden cells")
    
    @line_magic
    def llm_context_persist(self, line):
        """Toggle context persistence mode.
        
        When enabled (default), the kernel loads all notebook cells on startup.
        When disabled, only current session cells are used.
        
        Usage:
            %llm_context_persist          # Toggle
            %llm_context_persist on       # Enable
            %llm_context_persist off      # Disable
            %llm_context_persist status   # Check status
        """
        arg = line.strip().lower()
        
        if not arg:
            # Toggle
            self.kernel.context_persistence = not self.kernel.context_persistence
            status = "ON" if self.kernel.context_persistence else "OFF"
            print(f"📚 Context persistence: {status}")
        elif arg in ['on', 'true', '1']:
            self.kernel.context_persistence = True
            print("📚 Context persistence: ON")
            print("   Previous cells will be loaded automatically")
        elif arg in ['off', 'false', '0']:
            self.kernel.context_persistence = False
            print("📚 Context persistence: OFF")
            print("   Only current session cells will be used")
        elif arg == 'status':
            status = "ON" if self.kernel.context_persistence else "OFF"
            print(f"📚 Context persistence: {status}")
        else:
            print("Usage: %llm_context_persist [on|off|status]")
    
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
    def llm_rerank(self, line):
        """Rerank cells in context by relevance using LLM.
        
        Usage:
            %llm_rerank                    # Rerank based on last query
            %llm_rerank "specific query"   # Rerank based on specific query
            %llm_rerank --show             # Show ranking without reordering
            %llm_rerank --top=10           # Only keep top 10 most relevant
        """
        import json
        
        # Parse arguments
        show_only = '--show' in line
        top_n = None
        query = None
        
        # Extract top N if specified
        if '--top=' in line:
            import re
            match = re.search(r'--top=(\d+)', line)
            if match:
                top_n = int(match.group(1))
                line = re.sub(r'--top=\d+', '', line).strip()
        
        # Remove flags to get query
        line = line.replace('--show', '').strip()
        
        if line and not line.startswith('--'):
            query = line
        else:
            # Use the last user message as the query
            messages = self.kernel.get_notebook_cells_as_context()
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            if user_messages:
                query = user_messages[-1]['content']
            else:
                print("❌ No query provided and no previous user messages found")
                return
        
        print(f"🔍 Reranking cells based on: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        print("⏳ Asking LLM to analyze relevance...")
        
        # Get current context
        messages = self.kernel.get_notebook_cells_as_context()
        
        if not messages:
            print("❌ No cells in context to rerank")
            return
        
        # Create prompt for LLM to rank cells
        ranking_prompt = f"""Analyze the following conversation cells and rank them by relevance to this query:
"{query}"

Here are the cells to rank:

"""
        
        # Add numbered cells
        for i, msg in enumerate(messages):
            content_preview = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            ranking_prompt += f"[Cell {i+1}] {msg['role'].upper()}:\n{content_preview}\n\n"
        
        ranking_prompt += f"""
Please rank these cells from MOST to LEAST relevant to the query "{query}".
Return ONLY a JSON array of cell numbers in order of relevance.
Example: [3, 1, 5, 2, 4]

Your ranking:"""

        try:
            # Query the LLM for rankings
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, 
                        self.kernel.query_llm_async(ranking_prompt, temperature=0))
                    ranking_response = future.result()
            else:
                ranking_response = loop.run_until_complete(
                    self.kernel.query_llm_async(ranking_prompt, temperature=0)
                )
            
            # Parse the ranking
            import re
            json_match = re.search(r'\[[\d,\s]+\]', ranking_response)
            if json_match:
                rankings = json.loads(json_match.group())
                
                # Validate rankings
                valid_rankings = []
                for r in rankings:
                    if isinstance(r, int) and 1 <= r <= len(messages):
                        valid_rankings.append(r - 1)  # Convert to 0-based
                
                if not valid_rankings:
                    print("❌ LLM returned invalid rankings")
                    return
                
                # Apply top N filter if specified
                if top_n and len(valid_rankings) > top_n:
                    valid_rankings = valid_rankings[:top_n]
                    print(f"📊 Keeping top {top_n} most relevant cells")
                
                # Create reordered messages
                reordered_messages = []
                for idx in valid_rankings:
                    if idx < len(messages):
                        reordered_messages.append(messages[idx])
                
                # Add any messages not in the ranking (shouldn't happen but be safe)
                for i, msg in enumerate(messages):
                    if i not in valid_rankings:
                        reordered_messages.append(msg)
                
                if show_only:
                    # Just show the ranking
                    print("\n📊 Relevance Ranking:")
                    print("=" * 50)
                    for rank, idx in enumerate(valid_rankings):
                        msg = messages[idx]
                        content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                        print(f"{rank+1}. [Cell {idx+1}] {msg['role'].upper()}: {content_preview}")
                else:
                    # Actually reorder the context
                    # Store the reordered context
                    self.kernel._reranked_context = reordered_messages
                    
                    print(f"✅ Reranked {len(valid_rankings)} cells by relevance")
                    print("📝 Context has been reordered (most relevant first)")
                    print("💡 Use %llm_context to see the new order")
                    
            else:
                print("❌ Could not parse LLM ranking response")
                print(f"Response: {ranking_response}")
                
        except Exception as e:
            print(f"❌ Error during reranking: {e}")
    
    @line_magic
    def llm_apply_meta(self, line):
        """Apply custom meta functions to context.
        
        Usage:
            %llm_apply_meta ranking "query"    # Apply custom ranking
            %llm_apply_meta filter             # Apply custom filter
            %llm_apply_meta transform          # Apply custom transform
            %llm_apply_meta all               # Apply all in order: filter, ranking, transform
        """
        if not hasattr(self.kernel, '_meta_functions') or not self.kernel._meta_functions:
            print("❌ No meta functions defined. Use %%meta to define custom functions.")
            return
            
        args = line.split(None, 1)
        if not args:
            print("Usage: %llm_apply_meta <type> [query]")
            return
            
        meta_type = args[0].lower()
        query = args[1] if len(args) > 1 else None
        
        # Get current context
        messages = self.kernel.get_notebook_cells_as_context()
        
        if meta_type == 'all':
            # Apply all meta functions in order
            applied = []
            
            # 1. Filter
            if 'filter' in self.kernel._meta_functions:
                try:
                    messages = self.kernel._meta_functions['filter'](messages)
                    applied.append('filter')
                except Exception as e:
                    print(f"❌ Error in filter function: {e}")
                    
            # 2. Ranking (needs query)
            if 'ranking' in self.kernel._meta_functions and query:
                try:
                    messages = self.kernel._meta_functions['ranking'](messages, query)
                    applied.append('ranking')
                except Exception as e:
                    print(f"❌ Error in ranking function: {e}")
                    
            # 3. Transform
            if 'transform' in self.kernel._meta_functions:
                try:
                    messages = self.kernel._meta_functions['transform'](messages)
                    applied.append('transform')
                except Exception as e:
                    print(f"❌ Error in transform function: {e}")
                    
            if applied:
                self.kernel._reranked_context = messages
                print(f"✅ Applied meta functions: {', '.join(applied)}")
                print(f"📊 Context size: {len(messages)} messages")
            else:
                print("⚠️  No meta functions were applied")
                
        elif meta_type in self.kernel._meta_functions:
            # Apply specific meta function
            try:
                func = self.kernel._meta_functions[meta_type]
                
                if meta_type == 'ranking':
                    if not query:
                        print("❌ Ranking requires a query. Usage: %llm_apply_meta ranking \"your query\"")
                        return
                    result = func(messages, query)
                else:
                    result = func(messages)
                    
                self.kernel._reranked_context = result
                print(f"✅ Applied {meta_type} function")
                print(f"📊 Context size: {len(result)} messages")
                
            except Exception as e:
                print(f"❌ Error applying {meta_type} function: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ No {meta_type} function defined. Available: {list(self.kernel._meta_functions.keys())}")
    
    @line_magic
    def llm_meta_list(self, line):
        """List all defined meta functions."""
        if not hasattr(self.kernel, '_meta_functions') or not self.kernel._meta_functions:
            print("No meta functions defined")
            return
            
        print("📋 Defined meta functions:")
        for name, func in self.kernel._meta_functions.items():
            print(f"  - {name}: {func.__name__}")
            if hasattr(func, '__doc__') and func.__doc__:
                print(f"    {func.__doc__.strip()}")
    
    @line_magic
    def llm_rerank_clear(self, line):
        """Clear the reranked context and return to original order."""
        if hasattr(self.kernel, '_reranked_context'):
            del self.kernel._reranked_context
            print("✅ Cleared reranking - context restored to original order")
        else:
            print("ℹ️  No reranking active")
    
    @line_magic
    def llm_config(self, line):
        """Show interactive configuration panel."""
        self.create_config_panel()
    
    @line_magic
    def llm_context_window(self, line):
        """Display or set context window information for the current model.
        
        Usage:
            %llm_context_window           # Show current model's context window
            %llm_context_window all       # Show all models' context windows
            %llm_context_window gpt-4     # Show specific model's context window
        """
        if not litellm:
            print("❌ LiteLLM not available")
            return
            
        arg = line.strip()
        
        if not arg:
            # Show current model's context window
            if not self.kernel.active_model:
                print("❌ No active model selected")
                return
                
            try:
                max_tokens = litellm.get_max_tokens(self.kernel.active_model)
                
                # Get current context usage
                messages = self.kernel.get_notebook_cells_as_context()
                current_tokens = 0
                
                if messages:
                    # Use litellm's token counter
                    try:
                        current_tokens = litellm.token_counter(
                            model=self.kernel.active_model,
                            messages=messages
                        )
                    except:
                        # Fallback to character estimate
                        total_chars = sum(len(msg['content']) for msg in messages)
                        current_tokens = total_chars // 4
                
                # Calculate usage percentage
                usage_pct = (current_tokens / max_tokens * 100) if max_tokens > 0 else 0
                
                # Create visual progress bar
                bar_width = 40
                filled = int(bar_width * usage_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                print(f"📊 Context Window for {self.kernel.active_model}:")
                print(f"   Max tokens: {max_tokens:,}")
                print(f"   Current usage: {current_tokens:,} tokens ({usage_pct:.1f}%)")
                print(f"   [{bar}]")
                
                if usage_pct > 80:
                    print("   ⚠️  Warning: Context window nearly full!")
                elif usage_pct > 60:
                    print("   ⚡ Context usage is getting high")
                    
            except Exception as e:
                print(f"❌ Error getting context window info: {e}")
                
        elif arg == 'all':
            # Show all models' context windows
            print("📊 Context Windows for All Models:")
            print("=" * 60)
            
            model_info = []
            for model_name in self.kernel.llm_clients.keys():
                try:
                    max_tokens = litellm.get_max_tokens(model_name)
                    model_info.append((model_name, max_tokens))
                except:
                    model_info.append((model_name, "Unknown"))
            
            # Sort by token count (highest first)
            model_info.sort(key=lambda x: x[1] if isinstance(x[1], int) else 0, reverse=True)
            
            for model, tokens in model_info:
                active = " ✅" if model == self.kernel.active_model else ""
                if isinstance(tokens, int):
                    print(f"   {model:<30} {tokens:>10,} tokens{active}")
                else:
                    print(f"   {model:<30} {tokens:>10}{active}")
                    
        else:
            # Show specific model's context window
            try:
                max_tokens = litellm.get_max_tokens(arg)
                print(f"📊 Context Window for {arg}: {max_tokens:,} tokens")
            except Exception as e:
                print(f"❌ Error getting context window for {arg}: {e}")
    
    @line_magic
    def llm_token_count(self, line):
        """Count tokens in current context or provided text.
        
        Usage:
            %llm_token_count                    # Count tokens in current context
            %llm_token_count "some text"        # Count tokens in provided text
            %llm_token_count --model=gpt-4      # Use specific model's tokenizer
        """
        if not litellm:
            print("❌ LiteLLM not available")
            return
            
        # Parse arguments
        model = self.kernel.active_model
        text = None
        
        if '--model=' in line:
            import re
            match = re.search(r'--model=(\S+)', line)
            if match:
                model = match.group(1)
                line = re.sub(r'--model=\S+', '', line).strip()
        
        if line.strip():
            # Count tokens in provided text
            text = line.strip().strip('"\'')
        else:
            # Count tokens in current context
            messages = self.kernel.get_notebook_cells_as_context()
            if not messages:
                print("❌ No context available")
                return
        
        try:
            if text:
                # Count tokens in text
                tokens = litellm.token_counter(
                    model=model,
                    text=text
                )
                print(f"🔢 Token count for provided text:")
                print(f"   Model: {model}")
                print(f"   Tokens: {tokens:,}")
                print(f"   Characters: {len(text):,}")
                print(f"   Ratio: {len(text)/tokens:.2f} chars/token")
            else:
                # Count tokens in messages
                tokens = litellm.token_counter(
                    model=model,
                    messages=messages
                )
                
                # Get max tokens for comparison
                try:
                    max_tokens = litellm.get_max_tokens(model)
                    usage_pct = (tokens / max_tokens * 100) if max_tokens > 0 else 0
                    
                    print(f"🔢 Token count for current context:")
                    print(f"   Model: {model}")
                    print(f"   Messages: {len(messages)}")
                    print(f"   Tokens: {tokens:,} / {max_tokens:,} ({usage_pct:.1f}%)")
                except:
                    print(f"🔢 Token count for current context:")
                    print(f"   Model: {model}")
                    print(f"   Messages: {len(messages)}")
                    print(f"   Tokens: {tokens:,}")
                    
        except Exception as e:
            print(f"❌ Error counting tokens: {e}")
            # Fallback to character estimate
            if text:
                est_tokens = len(text) // 4
                print(f"   Estimated tokens (chars/4): {est_tokens:,}")
            else:
                total_chars = sum(len(msg['content']) for msg in messages)
                est_tokens = total_chars // 4
                print(f"   Estimated tokens (chars/4): {est_tokens:,}")
    
    @line_magic
    def llm_cost(self, line):
        """Show cost information for the current session or estimate costs.
        
        Usage:
            %llm_cost                    # Show session cost
            %llm_cost estimate           # Estimate cost for current context
            %llm_cost --model=gpt-4      # Show costs for specific model
        """
        if not litellm:
            print("❌ LiteLLM not available")
            return
            
        arg = line.strip()
        
        if not arg:
            # Show session costs
            if not hasattr(self.kernel, 'session_costs'):
                self.kernel.session_costs = {'total': 0.0, 'by_model': {}}
                
            print("💰 Session Costs:")
            print(f"   Total: ${self.kernel.session_costs['total']:.6f}")
            
            if self.kernel.session_costs['by_model']:
                print("\n   By Model:")
                for model, cost in self.kernel.session_costs['by_model'].items():
                    print(f"   - {model}: ${cost:.6f}")
            else:
                print("   No costs incurred yet")
                
        elif arg == 'estimate' or arg.startswith('--model='):
            # Estimate cost for current context
            model = self.kernel.active_model
            
            if arg.startswith('--model='):
                model = arg.split('=')[1]
                
            messages = self.kernel.get_notebook_cells_as_context()
            if not messages:
                print("❌ No context available")
                return
                
            try:
                # Count tokens
                prompt_tokens = litellm.token_counter(model=model, messages=messages)
                
                # Estimate completion tokens (rough estimate: 50% of prompt)
                completion_tokens = prompt_tokens // 2
                
                # Get costs
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                total_cost = prompt_cost + completion_cost
                
                print(f"💰 Estimated Cost for {model}:")
                print(f"   Prompt tokens: {prompt_tokens:,} (${prompt_cost:.6f})")
                print(f"   Est. completion tokens: {completion_tokens:,} (${completion_cost:.6f})")
                print(f"   Total estimate: ${total_cost:.6f}")
                
                # Show cost per 1K tokens
                if prompt_tokens > 0:
                    cost_per_1k_prompt = (prompt_cost / prompt_tokens) * 1000
                    cost_per_1k_completion = (completion_cost / completion_tokens) * 1000
                    print(f"\n   Cost per 1K tokens:")
                    print(f"   - Prompt: ${cost_per_1k_prompt:.6f}")
                    print(f"   - Completion: ${cost_per_1k_completion:.6f}")
                    
            except Exception as e:
                print(f"❌ Error estimating cost: {e}")
    
    @line_magic
    def llm_mcp_connect(self, line):
        """Connect to MCP servers.
        
        Usage:
            %llm_mcp_connect                    # Connect to all configured servers
            %llm_mcp_connect filesystem         # Connect to specific server
            %llm_mcp_connect https://example.com/mcp  # Connect to URL
        """
        import asyncio
        
        async def connect():
            if not line.strip():
                # Connect to all configured servers
                results = await self.kernel.mcp_manager.connect_all_servers()
                
                if not results:
                    print("❌ No MCP servers configured")
                    print("💡 Create a config file or use: %llm_mcp_connect <server_url>")
                    return
                
                print("🔌 MCP Server Connections:")
                for name, success in results.items():
                    status = "✅ Connected" if success else "❌ Failed"
                    print(f"   {name}: {status}")
                
                # Show available tools
                tools = self.kernel.mcp_manager.list_tools()
                if tools:
                    print(f"\n🛠️  {len(tools)} tools available")
                    print("   Use %llm_mcp_tools to list them")
            else:
                # Connect to specific server
                arg = line.strip()
                
                # Check if it's a configured server name
                if arg in self.kernel.mcp_manager.server_configs:
                    success = await self.kernel.mcp_manager.connect_server(
                        arg, self.kernel.mcp_manager.server_configs[arg]
                    )
                else:
                    # Treat as URL or path
                    server_name = arg.split('/')[-1].replace('.', '_')
                    success = await self.kernel.mcp_manager.connect_server(server_name, arg)
                
                if success:
                    print(f"✅ Connected to {arg}")
                    tools = [t for t in self.kernel.mcp_manager.list_tools() 
                            if t['server'] == (arg if arg in self.kernel.mcp_manager.server_configs else server_name)]
                    print(f"🛠️  {len(tools)} tools available")
                else:
                    print(f"❌ Failed to connect to {arg}")
        
        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, connect())
                future.result()
        else:
            loop.run_until_complete(connect())
    
    @line_magic
    def llm_mcp_disconnect(self, line):
        """Disconnect from MCP servers.
        
        Usage:
            %llm_mcp_disconnect              # Disconnect all
            %llm_mcp_disconnect filesystem   # Disconnect specific server
        """
        import asyncio
        
        async def disconnect():
            if not line.strip():
                # Disconnect all
                await self.kernel.mcp_manager.disconnect_all_servers()
                print("🔌 Disconnected from all MCP servers")
            else:
                # Disconnect specific server
                server_name = line.strip()
                success = await self.kernel.mcp_manager.disconnect_server(server_name)
                if success:
                    print(f"🔌 Disconnected from {server_name}")
                else:
                    print(f"❌ Server '{server_name}' not connected")
        
        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, disconnect())
                future.result()
        else:
            loop.run_until_complete(disconnect())
    
    @line_magic
    def llm_mcp_tools(self, line):
        """List available MCP tools.
        
        Usage:
            %llm_mcp_tools              # List all tools
            %llm_mcp_tools filesystem   # List tools from specific server
            %llm_mcp_tools --json       # Output as JSON
        """
        tools = self.kernel.mcp_manager.list_tools()
        
        if not tools:
            print("❌ No MCP tools available")
            print("💡 Connect to servers first: %llm_mcp_connect")
            return
        
        # Filter by server if specified
        server_filter = None
        output_json = False
        
        args = line.strip().split()
        for arg in args:
            if arg == '--json':
                output_json = True
            elif not arg.startswith('--'):
                server_filter = arg
        
        if server_filter:
            tools = [t for t in tools if t['server'] == server_filter]
        
        if output_json:
            import json
            print(json.dumps(tools, indent=2))
        else:
            # Group by server
            by_server = {}
            for tool in tools:
                server = tool['server']
                if server not in by_server:
                    by_server[server] = []
                by_server[server].append(tool)
            
            print(f"🛠️  Available MCP Tools ({len(tools)} total):")
            for server, server_tools in by_server.items():
                print(f"\n📦 {server} ({len(server_tools)} tools):")
                for tool in server_tools:
                    print(f"   • {tool['name']}")
                    if tool['description']:
                        print(f"     {tool['description']}")
    
    @line_magic
    def llm_mcp_call(self, line):
        """Call an MCP tool directly.
        
        Usage:
            %llm_mcp_call tool_name {"arg": "value"}
            %llm_mcp_call filesystem.read_file {"path": "/etc/hosts"}
        """
        import json
        import asyncio
        
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            print("Usage: %llm_mcp_call <tool_name> <json_arguments>")
            return
        
        tool_name = parts[0]
        
        try:
            arguments = json.loads(parts[1])
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON arguments: {e}")
            return
        
        async def call_tool():
            try:
                result = await self.kernel.mcp_manager.call_tool(tool_name, arguments)
                print(f"✅ Tool '{tool_name}' executed successfully:")
                
                # Pretty print the result
                if isinstance(result, dict) or isinstance(result, list):
                    print(json.dumps(result, indent=2))
                else:
                    print(result)
                    
            except ValueError as e:
                print(f"❌ {e}")
            except Exception as e:
                print(f"❌ Error calling tool: {e}")
        
        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, call_tool())
                future.result()
        else:
            loop.run_until_complete(call_tool())
    
    @line_magic
    def llm_mcp_config(self, line):
        """Show or set MCP configuration.
        
        Usage:
            %llm_mcp_config                    # Show current config
            %llm_mcp_config reload             # Reload config from file
            %llm_mcp_config path/to/config.json # Load specific config file
        """
        arg = line.strip()
        
        if not arg:
            # Show current configuration
            if not self.kernel.mcp_manager.server_configs:
                print("❌ No MCP servers configured")
                print("\n💡 Create a config file at one of these locations:")
                print("   - ~/.llm-kernel/mcp-config.json")
                print("   - ./mcp-config.json")
                print("   - ./.mcp/config.json")
                print("\nExample config:")
                print(json.dumps({
                    "mcpServers": {
                        "filesystem": {
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
                        },
                        "github": {
                            "url": "https://api.github.com/mcp"
                        }
                    }
                }, indent=2))
            else:
                print("📋 MCP Server Configuration:")
                print(json.dumps({"mcpServers": self.kernel.mcp_manager.server_configs}, indent=2))
                
                # Show connected servers
                connected = self.kernel.mcp_manager.get_connected_servers()
                if connected:
                    print(f"\n✅ Connected servers: {', '.join(connected)}")
                else:
                    print("\n⚠️  No servers connected. Use %llm_mcp_connect")
                    
        elif arg == 'reload':
            # Reload configuration
            old_config = self.kernel.mcp_manager.server_configs.copy()
            self.kernel.mcp_manager.load_config()
            
            if self.kernel.mcp_manager.server_configs != old_config:
                print("✅ Configuration reloaded")
                print("💡 Use %llm_mcp_connect to connect to new servers")
            else:
                print("ℹ️  Configuration unchanged")
                
        else:
            # Load specific config file
            try:
                self.kernel.mcp_manager.load_config(arg)
                print(f"✅ Loaded configuration from {arg}")
                print("💡 Use %llm_mcp_connect to connect to servers")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
    
    @cell_magic
    def llm_mcp(self, line, cell):
        """Query LLM with MCP tools available.
        
        The LLM will have access to all connected MCP tools and can use them
        to help answer your query.
        
        Usage:
            %%llm_mcp
            Can you read the README.md file and summarize it?
            
            %%llm_mcp --model=gpt-4
            Check the current git status and explain any uncommitted changes.
        """
        if not self.kernel.mcp_manager.is_connected():
            print("⚠️  No MCP servers connected. The LLM won't have access to external tools.")
            print("💡 Use %llm_mcp_connect to connect to MCP servers first")
        
        # TODO: Implement LLM querying with MCP tool access
        # This will require updating the LLM query to include available tools
        # and handle tool calls in the response
        
        # For now, just pass through to regular LLM query
        print("🔧 MCP-enhanced queries coming soon!")
        print("For now, using regular LLM query without tool access...")
        
        # Call the regular llm cell magic
        return self.llm(line, cell)

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
