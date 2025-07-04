"""
Base Magic Commands for LLM Kernel

Contains fundamental magic commands for chat mode, model management,
and basic LLM interaction.
"""

import time
import json
import asyncio
import concurrent.futures
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
from IPython.display import display, HTML, Markdown


@magics_class
class BaseMagics(Magics):
    """Base magic commands for LLM interaction."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
    
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
    def llm_models(self, line):
        """List available LLM models."""
        print("🤖 Available LLM Models:")
        for name, model_id in self.kernel.llm_clients.items():
            active = " ✅ (active)" if name == self.kernel.active_model else ""
            print(f"  - {name}: {model_id}{active}")
        
        if not self.kernel.llm_clients:
            print("  ❌ No models available. Check your API keys!")
    
    @line_magic
    def llm_model(self, line):
        """Switch active LLM model or show current model."""
        model = line.strip()
        
        if not model:
            # Show current model
            if self.kernel.active_model:
                print(f"Active model: {self.kernel.active_model}")
            else:
                print("No active model selected")
            return
        
        if model in self.kernel.llm_clients:
            self.kernel.active_model = model
            print(f"✅ Switched to {model}")
        else:
            print(f"❌ Model '{model}' not available")
            print("Available models:", ', '.join(self.kernel.llm_clients.keys()))
    
    @line_magic
    def llm_status(self, line):
        """Show comprehensive kernel status."""
        print("🤖 LLM Kernel Status")
        print("=" * 40)
        
        # Model status
        print(f"Active Model: {self.kernel.active_model or 'None'}")
        print(f"Available Models: {len(self.kernel.llm_clients)}")
        
        # Conversation history
        print(f"\nConversation History: {len(self.kernel.conversation_history)} exchanges")
        
        # Context window usage
        if hasattr(self.kernel, 'context_manager'):
            window_usage = self.kernel.context_manager.get_window_usage()
            print(f"Context Window Usage: {window_usage:.1f}%")
        
        # Token usage estimate
        total_tokens = sum(
            len(exchange.get('input', '')) + len(exchange.get('output', ''))
            for exchange in self.kernel.conversation_history
        ) // 4  # Rough token estimate
        
        print(f"Estimated Total Tokens: ~{total_tokens:,}")
        
        # Chat mode status
        chat_status = "ON" if getattr(self.kernel, 'chat_mode', False) else "OFF"
        print(f"\nChat Mode: {chat_status}")
        
        # Display mode
        display_mode = getattr(self.kernel, 'display_mode', 'inline')
        print(f"Display Mode: {display_mode}")
    
    @line_magic
    def llm_clear(self, line):
        """Clear conversation history."""
        self.kernel.conversation_history.clear()
        self.kernel.model_contexts.clear()
        if hasattr(self.kernel, 'context_manager'):
            self.kernel.context_manager.clear()
        print("✅ Conversation history cleared")
    
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
    
    @cell_magic
    def llm(self, line, cell):
        """Query LLM with cell content."""
        # Parse options from line
        import shlex
        try:
            args = shlex.split(line)
        except ValueError:
            args = line.split()
        
        model = None
        for i, arg in enumerate(args):
            if arg == '--model' and i + 1 < len(args):
                model = args[i + 1]
            elif arg.startswith('--model='):
                model = arg.split('=', 1)[1]
        
        query = cell.strip()
        if not query:
            print("❌ Please provide a query")
            return
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use thread pool for nested async
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.kernel.query_llm_async(query, model)
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self.kernel.query_llm_async(query, model)
                )
            
            # Display result based on mode
            if self.kernel.display_mode == 'chat':
                self._display_chat_response(result, model or self.kernel.active_model)
            else:
                display(Markdown(result))
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    @cell_magic
    def llm_gpt4(self, line, cell):
        """Shortcut for GPT-4."""
        return self.llm("--model=gpt-4", cell)
    
    @cell_magic  
    def llm_claude(self, line, cell):
        """Shortcut for Claude."""
        return self.llm("--model=claude-3-sonnet", cell)
    
    @cell_magic
    def llm_compare(self, line, cell):
        """Compare responses from multiple models."""
        models = line.strip().split() if line.strip() else ['gpt-4o', 'claude-3-sonnet']
        query = cell.strip()
        
        if not query:
            print("❌ Please provide a query")
            return
        
        print(f"🔄 Comparing {len(models)} models...")
        results = {}
        
        async def get_all_responses():
            tasks = []
            for model in models:
                if model in self.kernel.llm_clients:
                    tasks.append(self.kernel.query_llm_async(query, model))
                else:
                    print(f"⚠️  Model '{model}' not available")
            
            if tasks:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for model, response in zip(models, responses):
                    if isinstance(response, Exception):
                        results[model] = f"Error: {response}"
                    else:
                        results[model] = response
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, get_all_responses())
                    future.result()
            else:
                loop.run_until_complete(get_all_responses())
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            return
        
        # Display results side by side
        if results:
            html = '<div style="display: flex; gap: 20px;">'
            for model, response in results.items():
                html += f'''
                <div style="flex: 1; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                    <h3 style="margin-top: 0;">🤖 {model}</h3>
                    <div style="max-height: 400px; overflow-y: auto;">
                        {response.replace('\n', '<br>')}
                    </div>
                </div>
                '''
            html += '</div>'
            display(HTML(html))
    
    def _display_chat_response(self, response: str, model: str):
        """Display response in chat format."""
        html = f'''
        <div style="margin: 10px 0 20px 40px; padding: 10px; background: #f5f5f5; 
                    border-radius: 10px; border-left: 3px solid #2196F3;">
            <strong>🤖 {model}:</strong><br>
            <div style="margin-top: 8px; white-space: pre-wrap;">{response}</div>
        </div>
        '''
        display(HTML(html))