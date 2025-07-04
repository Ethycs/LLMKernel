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
    
    def get_notebook_cells_as_context(self, max_cells=50):
        """Get notebook cells as context for LLM.
        
        Returns a list of messages built from notebook cells where:
        - Code cells with no output are user messages
        - Code cells with output become user/assistant pairs
        - Markdown cells are system messages for context
        """
        messages = []
        
        # Try to get notebook cells from the shell's history
        try:
            # Get all cells from the current session
            if hasattr(self.shell, 'history_manager'):
                # Get recent cell inputs and outputs
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
                
                # Estimate tokens (rough approximation)
                total_chars = sum(len(msg['content']) for msg in messages)
                estimated_tokens = total_chars // 4  # Rough estimate
                print(f"Estimated tokens: ~{estimated_tokens}")
                
                # Show hidden cells if any
                if hasattr(self.kernel, 'hidden_cells') and self.kernel.hidden_cells:
                    hidden_nums = sorted([int(cell_id.split('_')[1]) for cell_id in self.kernel.hidden_cells])
                    print(f"\n🙈 Hidden cells: {', '.join(map(str, hidden_nums))}")
        else:
            print("📝 Traditional Context Mode")
            print("Use %llm_status to see conversation history")
    
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
