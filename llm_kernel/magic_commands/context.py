"""
Context-related Magic Commands for LLM Kernel

Contains magic commands for managing context, hiding cells,
and context persistence.
"""

import os
import time
import json
from pathlib import Path
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
from IPython.display import display, HTML

try:
    import litellm
except ImportError:
    litellm = None


@magics_class
class ContextMagics(Magics):
    """Context management magic commands."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
    
    @line_magic
    def llm_context(self, line):
        """Show current context that will be sent to the LLM.
        
        This command also rescans the notebook file to pick up any new or edited cells above.
        """
        # Parse arguments
        args = line.strip().split()
        no_rescan = '--no-rescan' in args
        
        # Always show notebook cells as context when in a notebook environment
        print("📓 Notebook Context - Showing cells that will be sent to LLM:")
        
        # Show notebook file info if available
        if hasattr(self.kernel, 'notebook_utils'):
            nb_path = self.kernel.notebook_utils.get_notebook_path()
            if nb_path:
                print(f"📔 Reading from: {nb_path}")
                if not no_rescan:
                    print("🔄 Rescanning notebook for changes...")
            else:
                print("📔 Notebook file not found - using execution history")
        
        print("=" * 60)
        
        # Track if we had cells since last scan
        cells_since_scan = getattr(self.kernel, '_cells_since_last_scan', 0)
        
        # Get context with rescan (unless --no-rescan is specified)
        messages = self.kernel.get_notebook_cells_as_context(force_rescan=not no_rescan)
        
        # Show if auto-rescan happened
        if cells_since_scan > 0 and not no_rescan:
            print(f"✨ Auto-rescanned ({cells_since_scan} new cells detected)")
        
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
                    
                # Show info about rescanning
                print(f"\n💡 Tip: %llm_context rescans the notebook for changes")
                print(f"   Use %llm_context --no-rescan to skip rescanning")
    
    
    @cell_magic
    def hide(self, line, cell):
        """Hide this cell from the LLM context.
        
        The cell will still execute normally but won't be included
        in the context sent to the LLM.
        """
        # Mark this cell as hidden
        if not hasattr(self.kernel, 'hidden_cells'):
            self.kernel.hidden_cells = set()
        
        # Get current cell ID
        cell_id = getattr(self.kernel, '_current_cell_id', f"cell_{len(self.kernel.execution_tracker.execution_history)}")
        self.kernel.hidden_cells.add(cell_id)
        
        # Also track in execution tracker
        if hasattr(self.kernel.execution_tracker, 'hidden_cells'):
            self.kernel.execution_tracker.hidden_cells.add(cell_id)
        
        # Execute the cell normally
        self.shell.run_cell(cell)
        
        # Show indicator that cell is hidden
        print("🙈 Cell hidden from LLM context")
    
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
    def llm_context_save(self, line):
        """Save current context to a file.
        
        Usage:
            %llm_context_save           # Save to auto-named file
            %llm_context_save my_context.json  # Save to specific file
        """
        filename = line.strip() or f"llm_context_{int(time.time())}.json"
        
        try:
            context_data = {
                'messages': self.kernel.get_notebook_cells_as_context(),
                'hidden_cells': list(getattr(self.kernel, 'hidden_cells', set())),
                'conversation_history': self.kernel.conversation_history,
                'metadata': {
                    'saved_at': time.time(),
                    'active_model': self.kernel.active_model
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(context_data, f, indent=2)
            
            print(f"✅ Context saved to {filename}")
            
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
            print("❌ Please specify a filename")
            return
        
        try:
            with open(filename, 'r') as f:
                context_data = json.load(f)
            
            # Restore conversation history
            self.kernel.conversation_history = context_data.get('conversation_history', [])
            
            # Restore hidden cells
            if 'hidden_cells' in context_data:
                self.kernel.hidden_cells = set(context_data['hidden_cells'])
                if hasattr(self.kernel.execution_tracker, 'hidden_cells'):
                    self.kernel.execution_tracker.hidden_cells = set(context_data['hidden_cells'])
            
            # Restore metadata
            metadata = context_data.get('metadata', {})
            if 'active_model' in metadata:
                self.kernel.active_model = metadata['active_model']
            
            # Store loaded context for later use
            self.kernel.saved_context = context_data.get('messages', [])
            
            print(f"✅ Context loaded from {filename}")
            print(f"   Messages: {len(context_data.get('messages', []))}")
            print(f"   Hidden cells: {len(context_data.get('hidden_cells', []))}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
        except Exception as e:
            print(f"❌ Error loading context: {e}")
    
    @line_magic
    def llm_context_reset(self, line):
        """Reset context to a clean state.
        
        Usage:
            %llm_context_reset           # Clear everything
            %llm_context_reset --keep-hidden  # Clear but keep hidden cells
        """
        keep_hidden = '--keep-hidden' in line
        
        # Clear conversation history
        self.kernel.conversation_history.clear()
        self.kernel.model_contexts.clear()
        
        # Clear context manager
        if hasattr(self.kernel, 'context_manager'):
            self.kernel.context_manager.clear()
        
        # Clear saved context
        self.kernel.saved_context = None
        
        # Clear reranked context if exists
        if hasattr(self.kernel, '_reranked_context'):
            del self.kernel._reranked_context
        
        if not keep_hidden:
            # Clear hidden cells
            if hasattr(self.kernel, 'hidden_cells'):
                self.kernel.hidden_cells.clear()
            if hasattr(self.kernel.execution_tracker, 'hidden_cells'):
                self.kernel.execution_tracker.hidden_cells.clear()
            print("✅ Context reset (including hidden cells)")
        else:
            print("✅ Context reset (hidden cells preserved)")
    
    @line_magic
    def llm_context_persist(self, line):
        """Control whether context persists across kernel restarts.
        
        Usage:
            %llm_context_persist          # Toggle
            %llm_context_persist on       # Enable persistence
            %llm_context_persist off      # Disable persistence
            %llm_context_persist status   # Check current setting
        """
        arg = line.strip().lower()
        
        if not arg:
            # Toggle
            self.kernel.context_persistence = not getattr(self.kernel, 'context_persistence', True)
            status = "ON" if self.kernel.context_persistence else "OFF"
            print(f"📁 Context persistence: {status}")
        elif arg in ['on', 'true', '1']:
            self.kernel.context_persistence = True
            print("📁 Context persistence: ON")
            print("💡 Notebook cells will be loaded as context on kernel restart")
        elif arg in ['off', 'false', '0']:
            self.kernel.context_persistence = False
            print("📁 Context persistence: OFF")
        elif arg == 'status':
            status = "ON" if getattr(self.kernel, 'context_persistence', True) else "OFF"
            print(f"📁 Context persistence: {status}")
        else:
            print("Usage: %llm_context_persist [on|off|status]")
    
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
    def llm_history(self, line):
        """Display conversation history with formatting."""
        # Parse arguments
        show_all = '--all' in line
        show_cells = '--cells' in line
        last_n = 10  # Default
        
        # Extract --last=N
        import re
        match = re.search(r'--last=(\d+)', line)
        if match:
            last_n = int(match.group(1))
        
        history = self.kernel.conversation_history.copy()
        
        if not history:
            print("No conversation history yet")
            return
        
        # Apply filters
        if not show_all and len(history) > last_n:
            history = history[-last_n:]
            print(f"Showing last {last_n} exchanges (use --all for complete history)")
        
        # Group by model if requested
        html_parts = ['<div style="font-family: monospace;">']
        
        # Sort by relevance if available
        if hasattr(self.kernel, '_relevance_scores') and '--by-relevance' in line:
            sorted_history = sorted(
                history,
                key=lambda x: self.kernel._relevance_scores.get(x.get('cell_id', ''), 0),
                reverse=True
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
    def llm_prune(self, line):
        """Prune conversation history intelligently."""
        strategy = 'smart'  # Default strategy
        
        # Parse arguments
        if '--strategy=' in line:
            import re
            match = re.search(r'--strategy=(\w+)', line)
            if match:
                strategy = match.group(1)
        
        original_count = len(self.kernel.conversation_history)
        
        if strategy == 'semantic':
            # Use dialogue pruner with semantic similarity
            threshold = 0.7
            if '--threshold=' in line:
                match = re.search(r'--threshold=([\d.]+)', line)
                if match:
                    threshold = float(match.group(1))
            
            pruned_history = self.kernel.dialogue_pruner.prune_by_similarity(
                self.kernel.conversation_history,
                threshold=threshold
            )
        elif strategy == 'recency':
            # Keep only recent exchanges
            keep_last = 10
            if '--keep=' in line:
                match = re.search(r'--keep=(\d+)', line)
                if match:
                    keep_last = int(match.group(1))
            
            pruned_history = self.kernel.conversation_history[-keep_last:]
        else:
            # Smart pruning (default)
            pruned_history = self.kernel.dialogue_pruner.prune_dialogue(
                self.kernel.conversation_history,
                self.kernel.active_model
            )
        
        # Update history
        self.kernel.conversation_history = pruned_history
        pruned_count = original_count - len(pruned_history)
        
        print(f"✂️ Pruned {pruned_count} exchanges using {strategy} strategy")
        print(f"📊 Context size: {original_count} → {len(pruned_history)} exchanges")