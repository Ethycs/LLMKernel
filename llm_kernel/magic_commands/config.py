"""
Configuration and Cost Management Magic Commands for LLM Kernel

Contains magic commands for kernel configuration, token counting,
and cost tracking.
"""

import re
import ipywidgets as widgets
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import display

try:
    import litellm
except ImportError:
    litellm = None


@magics_class
class ConfigMagics(Magics):
    """Configuration and cost management magic commands."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
    
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