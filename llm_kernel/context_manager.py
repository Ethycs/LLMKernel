"""
Context Management System

This module handles intelligent context window management for the LLM kernel,
including cell dependency tracking, execution order management, and context optimization.
"""

import ast
import time
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CellExecution:
    """Represents a single cell execution."""
    cell_id: str
    content: str
    execution_count: int
    timestamp: float
    variables_defined: Set[str]
    variables_used: Set[str]
    imports: Set[str]
    functions_defined: Set[str]


class ExecutionTracker:
    """Tracks cell execution order and dependencies."""
    
    def __init__(self):
        self.execution_history: List[CellExecution] = []
        self.cell_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.cell_metadata: Dict[str, Dict[str, Any]] = {}
        self.pinned_cells: Set[str] = set()
        self.ignored_cells: Set[str] = set()
        
    def track_execution(self, cell_id: str, cell_content: str, execution_count: int):
        """Track a cell execution and analyze its dependencies."""
        # Analyze the cell content
        variables_defined, variables_used, imports, functions_defined = self.analyze_cell_content(cell_content)
        
        # Create execution record
        execution = CellExecution(
            cell_id=cell_id,
            content=cell_content,
            execution_count=execution_count,
            timestamp=time.time(),
            variables_defined=variables_defined,
            variables_used=variables_used,
            imports=imports,
            functions_defined=functions_defined
        )
        
        self.execution_history.append(execution)
        
        # Update dependencies
        self.update_dependencies(cell_id, variables_used, functions_defined)
        
        # Store metadata
        self.cell_metadata[cell_id] = {
            'execution_count': execution_count,
            'timestamp': execution.timestamp,
            'content_length': len(cell_content),
            'has_imports': bool(imports),
            'defines_functions': bool(functions_defined),
            'defines_variables': bool(variables_defined)
        }
    
    def analyze_cell_content(self, content: str) -> tuple:
        """Analyze cell content to extract variables, imports, and functions."""
        variables_defined = set()
        variables_used = set()
        imports = set()
        functions_defined = set()
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Variable assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables_defined.add(target.id)
                        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    variables_defined.add(elt.id)
                
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    # Variable usage
                    variables_used.add(node.id)
                
                elif isinstance(node, ast.Import):
                    # Import statements
                    for alias in node.names:
                        imports.add(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    # From imports
                    if node.module:
                        imports.add(node.module)
                    for alias in node.names:
                        imports.add(alias.name)
                
                elif isinstance(node, ast.FunctionDef):
                    # Function definitions
                    functions_defined.add(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    # Class definitions (treat as variables)
                    variables_defined.add(node.name)
        
        except SyntaxError:
            # If we can't parse, just return empty sets
            pass
        
        return variables_defined, variables_used, imports, functions_defined
    
    def update_dependencies(self, cell_id: str, variables_used: Set[str], functions_used: Set[str]):
        """Update dependency graph based on variable and function usage."""
        dependencies = set()
        
        # Find cells that define variables/functions used by this cell
        for execution in self.execution_history:
            if execution.cell_id == cell_id:
                continue
                
            # Check if this cell defines something we use
            if (execution.variables_defined & variables_used or 
                execution.functions_defined & functions_used):
                dependencies.add(execution.cell_id)
        
        self.cell_dependencies[cell_id] = dependencies
    
    def get_dependencies(self, cell_id: str) -> Set[str]:
        """Get all dependencies for a cell."""
        return self.cell_dependencies.get(cell_id, set())
    
    def get_dependency_chain(self, cell_id: str) -> List[str]:
        """Get the full dependency chain for a cell."""
        visited = set()
        chain = []
        
        def _collect_deps(cid):
            if cid in visited:
                return
            visited.add(cid)
            
            for dep in self.get_dependencies(cid):
                _collect_deps(dep)
            
            if cid != cell_id:  # Don't include the target cell itself
                chain.append(cid)
        
        _collect_deps(cell_id)
        return chain
    
    def pin_cell(self, cell_id: str):
        """Mark a cell as always important for context."""
        self.pinned_cells.add(cell_id)
        self.ignored_cells.discard(cell_id)  # Remove from ignored if present
    
    def unpin_cell(self, cell_id: str):
        """Remove pin from a cell."""
        self.pinned_cells.discard(cell_id)
    
    def ignore_cell(self, cell_id: str):
        """Mark a cell to be ignored in context."""
        self.ignored_cells.add(cell_id)
        self.pinned_cells.discard(cell_id)  # Remove from pinned if present
    
    def include_cell(self, cell_id: str):
        """Remove ignore flag from a cell."""
        self.ignored_cells.discard(cell_id)


class ContextManager:
    """Manages context windows for LLM interactions."""
    
    def __init__(self):
        self.execution_tracker = ExecutionTracker()
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_strategy = 'smart'  # 'chronological', 'dependency', 'smart', 'manual'
        self.max_context_tokens = 4000
        self.max_context_cells = 20
    
    def add_exchange(self, exchange: Dict[str, Any]):
        """Add an LLM exchange to the conversation history."""
        self.conversation_history.append(exchange)
    
    def get_context_for_model(self, model: str, target_cell: str = None) -> List[Dict[str, Any]]:
        """Get context for a specific model and cell."""
        if self.context_strategy == 'chronological':
            return self.get_chronological_context()
        elif self.context_strategy == 'dependency':
            return self.get_dependency_context(target_cell)
        elif self.context_strategy == 'manual':
            return self.get_manual_context()
        else:  # smart
            return self.get_smart_context(target_cell)
    
    def get_chronological_context(self) -> List[Dict[str, Any]]:
        """Get context based on chronological order."""
        # Return most recent exchanges up to limits
        context = self.conversation_history[-self.max_context_cells:]
        return self.trim_context_by_tokens(context)
    
    def get_dependency_context(self, target_cell: str = None) -> List[Dict[str, Any]]:
        """Get context based on cell dependencies."""
        if not target_cell:
            return self.get_chronological_context()
        
        # Get dependency chain
        dep_chain = self.execution_tracker.get_dependency_chain(target_cell)
        
        # Include pinned cells
        important_cells = set(dep_chain) | self.execution_tracker.pinned_cells
        
        # Filter conversation history to relevant cells
        relevant_exchanges = []
        for exchange in self.conversation_history:
            cell_id = exchange.get('cell_id')
            if cell_id in important_cells or cell_id is None:  # Include non-cell exchanges
                relevant_exchanges.append(exchange)
        
        return self.trim_context_by_tokens(relevant_exchanges)
    
    def get_manual_context(self) -> List[Dict[str, Any]]:
        """Get manually curated context (pinned cells only)."""
        pinned_cells = self.execution_tracker.pinned_cells
        
        relevant_exchanges = []
        for exchange in self.conversation_history:
            cell_id = exchange.get('cell_id')
            if cell_id in pinned_cells or cell_id is None:
                relevant_exchanges.append(exchange)
        
        return self.trim_context_by_tokens(relevant_exchanges)
    
    def get_smart_context(self, target_cell: str = None) -> List[Dict[str, Any]]:
        """Get context using smart hybrid strategy."""
        # Combine dependency-based and recency-based selection
        context_exchanges = []
        
        # Always include pinned cells
        pinned_cells = self.execution_tracker.pinned_cells
        
        # Get dependency chain if target cell is specified
        dep_cells = set()
        if target_cell:
            dep_cells = set(self.execution_tracker.get_dependency_chain(target_cell))
        
        # Score and select exchanges
        scored_exchanges = []
        for exchange in self.conversation_history:
            cell_id = exchange.get('cell_id')
            
            # Skip ignored cells
            if cell_id in self.execution_tracker.ignored_cells:
                continue
            
            score = self.calculate_relevance_score(exchange, target_cell, pinned_cells, dep_cells)
            scored_exchanges.append((exchange, score))
        
        # Sort by score and select top exchanges
        scored_exchanges.sort(key=lambda x: x[1], reverse=True)
        
        # Take top exchanges up to limits
        selected = [ex for ex, score in scored_exchanges[:self.max_context_cells] if score > 0]
        
        return self.trim_context_by_tokens(selected)
    
    def calculate_relevance_score(self, exchange: Dict[str, Any], target_cell: str, 
                                 pinned_cells: Set[str], dep_cells: Set[str]) -> float:
        """Calculate relevance score for an exchange."""
        cell_id = exchange.get('cell_id')
        timestamp = exchange.get('timestamp', 0)
        
        score = 0.0
        
        # Pinned cells get highest priority
        if cell_id in pinned_cells:
            score += 10.0
        
        # Dependency cells get high priority
        if cell_id in dep_cells:
            score += 5.0
        
        # Recency bonus (more recent = higher score)
        if timestamp > 0:
            age_hours = (time.time() - timestamp) / 3600
            recency_score = max(0, 2.0 - (age_hours / 24))  # Decay over 24 hours
            score += recency_score
        
        # Content-based scoring
        content_length = len(exchange.get('input', '') + exchange.get('output', ''))
        if content_length > 100:  # Substantial content
            score += 1.0
        
        # Error resolution bonus
        if 'error' in exchange.get('output', '').lower():
            score += 0.5
        
        return score
    
    def trim_context_by_tokens(self, exchanges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim context to fit within token limits."""
        if not exchanges:
            return []
        
        # Estimate tokens (rough approximation)
        total_tokens = 0
        trimmed = []
        
        # Process in reverse order to keep most recent/relevant
        for exchange in reversed(exchanges):
            exchange_tokens = self.estimate_tokens(exchange)
            
            if total_tokens + exchange_tokens <= self.max_context_tokens:
                trimmed.insert(0, exchange)  # Insert at beginning to maintain order
                total_tokens += exchange_tokens
            else:
                break
        
        return trimmed
    
    def estimate_tokens(self, exchange: Dict[str, Any]) -> int:
        """Rough token estimation for an exchange."""
        text = exchange.get('input', '') + exchange.get('output', '')
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def set_context_strategy(self, strategy: str):
        """Set the context selection strategy."""
        valid_strategies = ['chronological', 'dependency', 'smart', 'manual']
        if strategy in valid_strategies:
            self.context_strategy = strategy
        else:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
    
    def set_context_limits(self, max_tokens: int = None, max_cells: int = None):
        """Set context window limits."""
        if max_tokens is not None:
            self.max_context_tokens = max_tokens
        if max_cells is not None:
            self.max_context_cells = max_cells
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about current context."""
        return {
            'strategy': self.context_strategy,
            'max_tokens': self.max_context_tokens,
            'max_cells': self.max_context_cells,
            'total_exchanges': len(self.conversation_history),
            'pinned_cells': len(self.execution_tracker.pinned_cells),
            'ignored_cells': len(self.execution_tracker.ignored_cells),
            'tracked_cells': len(self.execution_tracker.execution_history)
        }
