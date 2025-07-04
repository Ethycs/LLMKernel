"""
Reranking and Meta Function Magic Commands for LLM Kernel

Contains magic commands for cell reranking and custom context processing.
"""

import json
import asyncio
import concurrent.futures
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class


@magics_class
class RerankingMagics(Magics):
    """Reranking and meta function magic commands."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
    
    @line_magic
    def llm_rerank(self, line):
        """Rerank cells in context by relevance using LLM.
        
        Usage:
            %llm_rerank                    # Rerank based on last query
            %llm_rerank "specific query"   # Rerank based on specific query
            %llm_rerank --show             # Show ranking without reordering
            %llm_rerank --top=10           # Only keep top 10 most relevant
        """
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
            ranking_response = self._run_async(
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
    def llm_rerank_clear(self, line):
        """Clear the reranked context and return to original order."""
        if hasattr(self.kernel, '_reranked_context'):
            del self.kernel._reranked_context
            print("✅ Cleared reranking - context restored to original order")
        else:
            print("ℹ️  No reranking active")
    
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
            return
            
        print(f"⚠️  No function found in cell. Make sure to define a function.")
    
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
    
    def _run_async(self, coro):
        """Run async function in sync context."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)