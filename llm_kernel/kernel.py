"""
Main LLM Kernel Implementation (Refactored)

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
from .llm_integration import LLMIntegration
from .notebook_utils import NotebookUtils

# Import magic command modules
from .magic_commands import (
    BaseMagics,
    ContextMagics,
    MCPMagics,
    RerankingMagics,
    ConfigMagics
)

try:
    from .magic_commands.multimodal import MultimodalMagics
    from .magic_commands.multimodal_native_pdf import NativePDFMagics
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False
    MultimodalMagics = None
    NativePDFMagics = None


class LLMKernel(IPythonKernel):
    """
    Custom Jupyter kernel with LiteLLM integration and context management.
    
    Features:
    - Multi-LLM provider support via LiteLLM
    - Intelligent context window management
    - Cell dependency tracking
    - Automatic dialogue pruning
    - Magic command interface
    - MCP (Model Context Protocol) integration
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
        
        # Hidden cells tracking
        self.hidden_cells = set()
        
        # Chat mode
        self.chat_mode = False
        
        # Track context scanning
        self._last_context_scan_cell = None  # Track which cell we last scanned at
        self._cells_since_last_scan = 0  # Count cells executed since last scan
        
        # Async executor for parallel queries
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # MCP (Model Context Protocol) manager
        try:
            self.mcp_manager = MCPManager(logger=self.log)
            self.mcp_manager.load_config()
        except Exception as e:
            self.log.warning(f"MCP manager initialization failed: {e}")
            self.mcp_manager = None
        
        # LLM integration handler
        self.llm_integration = LLMIntegration(self)
        
        # Notebook utilities for reading notebook files
        self.notebook_utils = NotebookUtils(self)
        
        # Multimodal content support
        try:
            from .multimodal import MultimodalContent
            self.multimodal = MultimodalContent(self)
        except ImportError:
            self.multimodal = None
        
        # Register magic commands
        self.register_magic_commands()
        
        self.log.info("LLM Kernel initialized successfully")
        self.log.info(f"Active model: {self.active_model}")
        self.log.info(f"Available models: {list(self.llm_clients.keys())}")

    def setup_logging(self):
        """Configure logging for the kernel."""
        # Check if logging is disabled
        if os.getenv('LLM_KERNEL_LOGGING', 'true').lower() == 'false':
            # Create a null logger that does nothing
            self.log = logging.getLogger('llm_kernel')
            self.log.setLevel(logging.CRITICAL + 1)  # Disable all logging
            self.log.handlers = []
            self.log.addHandler(logging.NullHandler())
            return
        
        log_level = os.getenv('LLM_KERNEL_DEBUG', 'INFO').upper()
        
        # Create a logger
        self.log = logging.getLogger('llm_kernel')
        self.log.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Remove any existing handlers to avoid duplicates
        self.log.handlers = []
        
        # File handler for debug logging
        log_file = os.getenv('LLM_KERNEL_LOG_FILE', 'llm_kernel.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.log.addHandler(file_handler)
        
        # Log startup
        self.log.info("="*60)
        self.log.info("LLM Kernel starting up...")
    
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
                'gpt-3.5-turbo': 'gpt-3.5-turbo',
                'gpt-4.1': 'gpt-4.1',  # New GPT-4.1 model
                'o3': 'o3',  # New o3 model
                'o3-mini': 'o3-mini'  # Also add o3-mini if available
            })
            
        if os.getenv('ANTHROPIC_API_KEY'):
            available_models.update({
                'claude-3-opus': 'claude-3-opus-20240229',
                'claude-3-sonnet': 'claude-3-sonnet-20240229',
                'claude-3-haiku': 'claude-3-haiku-20240307',
                'claude-opus-4': 'claude-opus-4-20250514',  # New Claude Opus 4
                'claude-sonnet-4': 'claude-sonnet-4-20250514',  # New Claude Sonnet 4
                'claude-3-5-sonnet': 'claude-3-5-sonnet-20241022'  # Claude 3.5 Sonnet
            })
            
        if os.getenv('GOOGLE_API_KEY'):
            available_models.update({
                'gemini-2.5-pro': 'gemini/gemini-2.5-pro'  # New Gemini 2.5 Pro
            })
            
        # Local models via Ollama
        if os.getenv('OLLAMA_HOST') or self.check_ollama():
            available_models.update({
                'ollama/llama3': 'ollama/llama3',
                'ollama/codellama': 'ollama/codellama',
                'ollama/mistral': 'ollama/mistral'
            })
        
        self.llm_clients = available_models
        
        # Set default active model
        if available_models and not self.active_model:
            # Prefer GPT-4 if available, otherwise first available
            if 'gpt-4o' in available_models:
                self.active_model = 'gpt-4o'
            else:
                self.active_model = list(available_models.keys())[0]
            
        self.log.info(f"Initialized {len(available_models)} LLM models")
        if self.active_model:
            self.log.info(f"Active model: {self.active_model}")

    def check_ollama(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=1)
            return response.status_code == 200
        except:
            return False

    def register_magic_commands(self):
        """Register all magic command classes."""
        self.log.debug("Starting magic command registration...")
        try:
            # Create instances of magic command classes
            self.log.debug("Creating magic command instances...")
            base_magics = BaseMagics(self.shell, self)
            context_magics = ContextMagics(self.shell, self)
            mcp_magics = MCPMagics(self.shell, self)
            reranking_magics = RerankingMagics(self.shell, self)
            config_magics = ConfigMagics(self.shell, self)
            self.log.debug("Magic command instances created successfully")
            
            # Register them with IPython
            self.shell.register_magic_function(base_magics.llm_chat, 'line', 'llm_chat')
            self.shell.register_magic_function(base_magics.llm_models, 'line', 'llm_models')
            self.shell.register_magic_function(base_magics.llm_model, 'line', 'llm_model')
            self.shell.register_magic_function(base_magics.llm_status, 'line', 'llm_status')
            self.shell.register_magic_function(base_magics.llm_clear, 'line', 'llm_clear')
            self.shell.register_magic_function(base_magics.llm_display, 'line', 'llm_display')
            self.shell.register_magic_function(base_magics.llm_debug, 'line', 'llm_debug')
            self.shell.register_magic_function(base_magics.llm, 'cell', 'llm')
            self.shell.register_magic_function(base_magics.llm_gpt4, 'cell', 'llm_gpt4')
            self.shell.register_magic_function(base_magics.llm_claude, 'cell', 'llm_claude')
            self.shell.register_magic_function(base_magics.llm_compare, 'cell', 'llm_compare')
            
            # Context commands
            self.shell.register_magic_function(context_magics.llm_context, 'line', 'llm_context')
            self.shell.register_magic_function(context_magics.hide, 'cell', 'hide')
            self.shell.register_magic_function(context_magics.llm_unhide, 'line', 'llm_unhide')
            self.shell.register_magic_function(context_magics.llm_hidden, 'line', 'llm_hidden')
            self.shell.register_magic_function(context_magics.llm_context_save, 'line', 'llm_context_save')
            self.shell.register_magic_function(context_magics.llm_context_load, 'line', 'llm_context_load')
            self.shell.register_magic_function(context_magics.llm_context_reset, 'line', 'llm_context_reset')
            self.shell.register_magic_function(context_magics.llm_context_persist, 'line', 'llm_context_persist')
            self.shell.register_magic_function(context_magics.llm_pin_cell, 'line', 'llm_pin_cell')
            self.shell.register_magic_function(context_magics.llm_unpin_cell, 'line', 'llm_unpin_cell')
            self.shell.register_magic_function(context_magics.llm_history, 'line', 'llm_history')
            self.shell.register_magic_function(context_magics.llm_prune, 'line', 'llm_prune')
            
            # MCP commands
            self.shell.register_magic_function(mcp_magics.llm_mcp_connect, 'line', 'llm_mcp_connect')
            self.shell.register_magic_function(mcp_magics.llm_mcp_disconnect, 'line', 'llm_mcp_disconnect')
            self.shell.register_magic_function(mcp_magics.llm_mcp_tools, 'line', 'llm_mcp_tools')
            self.shell.register_magic_function(mcp_magics.llm_mcp_call, 'line', 'llm_mcp_call')
            self.shell.register_magic_function(mcp_magics.llm_mcp_config, 'line', 'llm_mcp_config')
            self.shell.register_magic_function(mcp_magics.llm_mcp, 'cell', 'llm_mcp')
            
            # Reranking commands
            self.shell.register_magic_function(reranking_magics.llm_rerank, 'line', 'llm_rerank')
            self.shell.register_magic_function(reranking_magics.llm_rerank_clear, 'line', 'llm_rerank_clear')
            self.shell.register_magic_function(reranking_magics.llm_rerank_apply, 'line', 'llm_rerank_apply')
            self.shell.register_magic_function(reranking_magics.meta, 'cell', 'meta')
            self.shell.register_magic_function(reranking_magics.llm_apply_meta, 'line', 'llm_apply_meta')
            self.shell.register_magic_function(reranking_magics.llm_meta_list, 'line', 'llm_meta_list')
            
            # Config commands
            self.shell.register_magic_function(config_magics.llm_config, 'line', 'llm_config')
            self.shell.register_magic_function(config_magics.llm_context_window, 'line', 'llm_context_window')
            self.shell.register_magic_function(config_magics.llm_token_count, 'line', 'llm_token_count')
            self.shell.register_magic_function(config_magics.llm_cost, 'line', 'llm_cost')
            
            # Multimodal commands (if available)
            if HAS_MULTIMODAL and MultimodalMagics:
                multimodal_magics = MultimodalMagics(self.shell, self)
                self.shell.register_magic_function(multimodal_magics.llm_paste, 'line', 'llm_paste')
                self.shell.register_magic_function(multimodal_magics.llm_image, 'line', 'llm_image')
                self.shell.register_magic_function(multimodal_magics.llm_pdf, 'line', 'llm_pdf')
                self.shell.register_magic_function(multimodal_magics.llm_media_clear, 'line', 'llm_media_clear')
                self.shell.register_magic_function(multimodal_magics.llm_media_list, 'line', 'llm_media_list')
                self.shell.register_magic_function(multimodal_magics.llm_vision, 'cell', 'llm_vision')
                # Cache management commands
                self.shell.register_magic_function(multimodal_magics.llm_cache_info, 'line', 'llm_cache_info')
                self.shell.register_magic_function(multimodal_magics.llm_cache_list, 'line', 'llm_cache_list')
                self.shell.register_magic_function(multimodal_magics.llm_cache_clear, 'line', 'llm_cache_clear')
                self.log.info("Multimodal magic commands registered")
                
                # Native PDF commands (for direct PDF upload)
                if NativePDFMagics:
                    native_pdf_magics = NativePDFMagics(self.shell, self)
                    self.shell.register_magic_function(native_pdf_magics.llm_pdf_native, 'line', 'llm_pdf_native')
                    self.shell.register_magic_function(native_pdf_magics.llm_files_list, 'line', 'llm_files_list')
                    self.shell.register_magic_function(native_pdf_magics.llm_files_clear, 'line', 'llm_files_clear')
                    self.log.info("Native PDF magic commands registered")
            
            self.log.info("All magic commands registered successfully")
            
        except Exception as e:
            self.log.error(f"Error registering magic commands: {e}")
            import traceback
            traceback.print_exc()

    def get_notebook_cells_as_context(self, force_rescan: bool = False, auto_rescan: bool = True) -> List[Dict[str, str]]:
        """Get notebook cells as context messages for the LLM.
        
        Args:
            force_rescan: Force a rescan of the notebook file
            auto_rescan: Automatically rescan if cells were executed since last scan
        """
        # Determine if we should rescan
        should_rescan = force_rescan
        if auto_rescan and self._cells_since_last_scan > 0:
            self.log.debug(f"Auto-rescanning: {self._cells_since_last_scan} cells executed since last scan")
            should_rescan = True
        
        # Check if we have reranked context (unless rescanning)
        if not should_rescan and hasattr(self, '_reranked_context') and self._reranked_context:
            return self._reranked_context
            
        # If we have saved/loaded context, use that (unless rescanning)
        if not should_rescan and self.saved_context:
            return self.saved_context
        
        # Try to get cells from the notebook file first
        if hasattr(self, 'notebook_utils'):
            # Force reload the notebook if rescanning
            if should_rescan:
                self.notebook_utils.read_notebook(force_reload=True)
                self._last_context_scan_cell = self._current_cell_id
                self._cells_since_last_scan = 0
            
            # Determine up_to_cell based on current execution
            up_to_cell = None
            if hasattr(self, '_current_cell_content'):
                # Find the current cell index in the notebook
                cell_idx = self.notebook_utils.find_cell_index(self._current_cell_content)
                if cell_idx is not None:
                    # Include cells up to and including the current one
                    up_to_cell = cell_idx + 1
                    self.log.debug(f"Limiting context to cells up to index {cell_idx}")
            
            notebook_messages = self.notebook_utils.get_cells_as_context(
                include_outputs=True,
                skip_empty=True,
                up_to_cell=up_to_cell
            )
            if notebook_messages:
                # Apply hidden cells filter
                filtered_messages = []
                cell_count = 0
                
                for msg in notebook_messages:
                    cell_id = f"cell_{cell_count}"
                    if msg['role'] == 'user':  # Each user message is a new cell
                        if cell_id not in self.hidden_cells:
                            filtered_messages.append(msg)
                        cell_count += 1
                    else:  # Assistant messages (outputs) belong to the same cell
                        if f"cell_{cell_count-1}" not in self.hidden_cells:
                            filtered_messages.append(msg)
                
                return filtered_messages
        
        # Fallback to execution history if notebook reading fails
        messages = []
        try:
            # Get notebook history through ipykernel mechanisms
            history = self.shell.history_manager.get_range(output=True)
            
            cell_count = 0
            for session, line_num, (input_text, output) in history:
                # Skip if cell is hidden
                cell_id = f"cell_{cell_count}"
                if cell_id in self.hidden_cells:
                    cell_count += 1
                    continue
                    
                # Skip magic commands except %%llm variants
                if input_text.strip().startswith('%') and not input_text.strip().startswith('%%llm'):
                    cell_count += 1
                    continue
                
                # Add user message (input)
                if input_text.strip():
                    # Clean up cell magics
                    if input_text.startswith('%%llm'):
                        lines = input_text.split('\n', 1)
                        if len(lines) > 1:
                            input_text = lines[1]
                        else:
                            cell_count += 1
                            continue
                    
                    # Skip hide magic
                    if input_text.strip().startswith('%%hide'):
                        cell_count += 1
                        continue
                    
                    messages.append({"role": "user", "content": input_text.strip()})
                    
                    # Add assistant message (output) if available
                    if output:
                        # Clean output - remove prompts and cell execution indicators
                        cleaned_lines = []
                        for line in output.split('\n'):
                            # Skip prompt-like lines and system messages
                            if (line.startswith('Out[') or 
                                line.startswith('In [') or
                                line.strip() == '' or
                                line.startswith('✅') or
                                line.startswith('❌') or
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
        
        # Add conversation history (including pasted images) to the end
        if hasattr(self, 'conversation_history') and self.conversation_history:
            messages.extend(self.conversation_history)
            
        return messages

    async def query_llm_async(self, query: str, model: str = None, **kwargs) -> str:
        """Asynchronously query an LLM model."""
        return await self.llm_integration.query_llm_async(query, model, **kwargs)
    
    async def query_llm_with_mcp_async(self, query: str, model: str = None, **kwargs) -> str:
        """Query LLM with MCP tools available."""
        return await self.llm_integration.query_llm_with_mcp_async(query, model, **kwargs)

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
        """Override execute to track cell execution and handle chat mode."""
        # Track cell execution
        cell_id = f"cell_{len(self.execution_tracker.execution_history)}"
        self._current_cell_id = cell_id
        self._current_cell_content = code
        
        # Only count non-magic cells towards rescan trigger
        # (unless it's %llm_context which resets the counter)
        if not code.strip().startswith('%') or code.strip().startswith('%llm_context'):
            self._cells_since_last_scan += 1
        
        # Get execution count
        execution_count = self.execution_count if hasattr(self, 'execution_count') else 0
        self.execution_tracker.track_execution(cell_id, code, execution_count)
        
        # Check if we're in chat mode and this is not a magic command or shell command
        # Magic commands (%) and shell commands (!) should always execute normally
        code_stripped = code.strip()
        
        # Check if any line in the code starts with a magic command
        # This handles multi-line cells where magic commands aren't on the first line
        has_magic = any(line.strip().startswith('%') or line.strip().startswith('!') 
                       for line in code.splitlines())
        
        # Debug logging
        self.log.debug(f"do_execute called with code: {code_stripped[:50]}...")
        self.log.debug(f"Chat mode: {getattr(self, 'chat_mode', False)}")
        self.log.debug(f"Has magic command: {has_magic}")
        
        if (hasattr(self, 'chat_mode') and self.chat_mode and not has_magic):
            
            # In chat mode, treat non-magic/non-shell commands as LLM queries
            query = code_stripped
            if query:
                try:
                    # Use the LLM integration to handle the query
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Use nest_asyncio to handle nested event loops
                        import nest_asyncio
                        nest_asyncio.apply()
                        result = loop.run_until_complete(
                            self.query_llm_async(query)
                        )
                    else:
                        result = loop.run_until_complete(
                            self.query_llm_async(query)
                        )
                    
                    # Display result based on mode
                    if result:  # Only display if we have a result
                        if self.display_mode == 'chat':
                            # Use print with formatting instead of HTML display
                            print(f"\n🤖 {self.active_model}:")
                            print("-" * 40)
                            print(result)
                            print("-" * 40)
                        else:
                            # For markdown mode, just print the result
                            print(result)
                    else:
                        print(f"⚠️ No response from {self.active_model}")
                    
                    # Return success
                    return {'status': 'ok', 'execution_count': self.execution_count,
                            'payload': [], 'user_expressions': {}}
                    
                except Exception as e:
                    # On error, fall through to normal execution
                    self.log.error(f"Chat mode error: {e}")
        
        # Normal execution
        self.log.debug(f"Executing normally (not as LLM query): {code_stripped[:50]}...")
        result = super().do_execute(code, silent, store_history, user_expressions, allow_stdin)
        self.log.debug(f"Execution result: {result}")
        return result

    def do_shutdown(self, restart):
        """Clean shutdown of kernel."""
        # Disconnect from MCP servers
        if hasattr(self, 'mcp_manager') and self.mcp_manager:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.mcp_manager.disconnect_all_servers())
                else:
                    loop.run_until_complete(self.mcp_manager.disconnect_all_servers())
            except:
                pass
        
        # Clean up OpenAI assistants
        if hasattr(self, 'llm_integration') and hasattr(self.llm_integration, 'openai_assistant'):
            try:
                self.llm_integration.openai_assistant.cleanup()
            except:
                pass
        
        # Shutdown executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        return super().do_shutdown(restart)