"""
Dialogue Pruning System

This module implements intelligent dialogue pruning strategies to optimize
context windows by removing irrelevant or redundant conversation history.
"""

import time
import math
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict


class DialoguePruner:
    """Intelligent dialogue pruning for context optimization."""
    
    def __init__(self):
        self.relevance_threshold = 0.7
        self.pruning_strategies = {
            'semantic': self.semantic_pruning,
            'recency': self.recency_pruning,
            'dependency': self.dependency_pruning,
            'hybrid': self.hybrid_pruning
        }
    
    def prune_dialogue(self, conversation_history: List[Dict[str, Any]], 
                      strategy: str = 'hybrid', **kwargs) -> List[Dict[str, Any]]:
        """Prune dialogue using the specified strategy."""
        if strategy not in self.pruning_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.pruning_strategies.keys())}")
        
        if not conversation_history:
            return []
        
        return self.pruning_strategies[strategy](conversation_history, **kwargs)
    
    def semantic_pruning(self, conversation_history: List[Dict[str, Any]], 
                        current_query: str = None, threshold: float = None) -> List[Dict[str, Any]]:
        """Remove semantically irrelevant dialogue."""
        if threshold is None:
            threshold = self.relevance_threshold
        
        if not current_query:
            # If no current query, use recency-based pruning
            return self.recency_pruning(conversation_history)
        
        # Simple semantic similarity based on keyword overlap
        # In a full implementation, you'd use embeddings here
        query_words = set(current_query.lower().split())
        
        relevant_exchanges = []
        for exchange in conversation_history:
            similarity = self.calculate_keyword_similarity(exchange, query_words)
            
            if similarity > threshold:
                exchange_copy = exchange.copy()
                exchange_copy['relevance_score'] = similarity
                relevant_exchanges.append(exchange_copy)
        
        return relevant_exchanges
    
    def calculate_keyword_similarity(self, exchange: Dict[str, Any], query_words: Set[str]) -> float:
        """Calculate keyword-based similarity score."""
        exchange_text = (exchange.get('input', '') + ' ' + exchange.get('output', '')).lower()
        exchange_words = set(exchange_text.split())
        
        if not exchange_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & exchange_words
        union = query_words | exchange_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def recency_pruning(self, conversation_history: List[Dict[str, Any]], 
                       max_age_hours: float = 24.0) -> List[Dict[str, Any]]:
        """Remove old dialogue based on recency."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        recent_exchanges = []
        for exchange in conversation_history:
            timestamp = exchange.get('timestamp', current_time)
            
            if timestamp >= cutoff_time:
                # Add recency score
                age_hours = (current_time - timestamp) / 3600
                recency_score = max(0, 1.0 - (age_hours / max_age_hours))
                
                exchange_copy = exchange.copy()
                exchange_copy['relevance_score'] = recency_score
                recent_exchanges.append(exchange_copy)
        
        return recent_exchanges
    
    def dependency_pruning(self, conversation_history: List[Dict[str, Any]], 
                          current_cell: str = None, execution_tracker=None) -> List[Dict[str, Any]]:
        """Keep only exchanges that define variables/functions used in current context."""
        if not current_cell or not execution_tracker:
            # Fallback to recency pruning
            return self.recency_pruning(conversation_history)
        
        # Get dependency chain
        try:
            dep_chain = execution_tracker.get_dependency_chain(current_cell)
            pinned_cells = execution_tracker.pinned_cells
            important_cells = set(dep_chain) | pinned_cells
        except:
            # If execution tracker fails, use recency
            return self.recency_pruning(conversation_history)
        
        relevant_exchanges = []
        for exchange in conversation_history:
            cell_id = exchange.get('cell_id')
            
            if cell_id in important_cells or cell_id is None:
                exchange_copy = exchange.copy()
                # Higher score for pinned cells
                if cell_id in pinned_cells:
                    exchange_copy['relevance_score'] = 1.0
                elif cell_id in dep_chain:
                    exchange_copy['relevance_score'] = 0.8
                else:
                    exchange_copy['relevance_score'] = 0.5
                
                relevant_exchanges.append(exchange_copy)
        
        return relevant_exchanges
    
    def hybrid_pruning(self, conversation_history: List[Dict[str, Any]], 
                      current_query: str = None, current_cell: str = None,
                      execution_tracker=None, context_budget: int = 4000,
                      threshold: float = None) -> List[Dict[str, Any]]:
        """Intelligent pruning using multiple criteria."""
        if threshold is None:
            threshold = self.relevance_threshold
        
        # Score each exchange on multiple dimensions
        scored_exchanges = []
        for exchange in conversation_history:
            score = self.calculate_multi_dimensional_score(
                exchange, current_query, current_cell, execution_tracker
            )
            
            if score > 0:  # Only include exchanges with positive scores
                exchange_copy = exchange.copy()
                exchange_copy['relevance_score'] = score
                scored_exchanges.append((exchange_copy, score))
        
        # Sort by relevance score
        scored_exchanges.sort(key=lambda x: x[1], reverse=True)
        
        # Add exchanges until we hit token budget
        selected_exchanges = []
        current_tokens = 0
        
        for exchange, score in scored_exchanges:
            if score < threshold:
                break
                
            exchange_tokens = self.estimate_tokens(exchange)
            
            if current_tokens + exchange_tokens <= context_budget:
                selected_exchanges.append(exchange)
                current_tokens += exchange_tokens
            else:
                # Try to summarize if it's important enough
                if score > 0.8:
                    summary = self.summarize_exchange(exchange)
                    summary_tokens = self.estimate_tokens(summary)
                    
                    if current_tokens + summary_tokens <= context_budget:
                        selected_exchanges.append(summary)
                        current_tokens += summary_tokens
        
        return selected_exchanges
    
    def calculate_multi_dimensional_score(self, exchange: Dict[str, Any], 
                                        current_query: str = None,
                                        current_cell: str = None,
                                        execution_tracker=None) -> float:
        """Calculate multi-factor relevance score."""
        factors = {}
        
        # Semantic similarity factor
        if current_query:
            query_words = set(current_query.lower().split())
            factors['semantic_similarity'] = self.calculate_keyword_similarity(exchange, query_words)
        else:
            factors['semantic_similarity'] = 0.5  # Neutral score
        
        # Recency factor
        factors['recency'] = self.calculate_recency_score(exchange)
        
        # Code dependency factor
        factors['code_dependency'] = self.calculate_dependency_score(
            exchange, current_cell, execution_tracker
        )
        
        # User importance factor (pinned cells, etc.)
        factors['user_importance'] = self.calculate_user_importance_score(
            exchange, execution_tracker
        )
        
        # Error resolution factor
        factors['error_resolution'] = self.calculate_error_resolution_score(exchange)
        
        # Content quality factor
        factors['content_quality'] = self.calculate_content_quality_score(exchange)
        
        # Weighted combination
        weights = {
            'semantic_similarity': 0.25,
            'recency': 0.20,
            'code_dependency': 0.25,
            'user_importance': 0.15,
            'error_resolution': 0.10,
            'content_quality': 0.05
        }
        
        total_score = sum(factors[k] * weights[k] for k in factors if k in weights)
        return max(0.0, min(1.0, total_score))  # Clamp to [0, 1]
    
    def calculate_recency_score(self, exchange: Dict[str, Any]) -> float:
        """Calculate recency-based score."""
        timestamp = exchange.get('timestamp', 0)
        if timestamp == 0:
            return 0.5  # Neutral score for unknown timestamps
        
        age_hours = (time.time() - timestamp) / 3600
        # Exponential decay over 48 hours
        return math.exp(-age_hours / 24.0)
    
    def calculate_dependency_score(self, exchange: Dict[str, Any], 
                                 current_cell: str = None,
                                 execution_tracker=None) -> float:
        """Calculate dependency-based score."""
        if not current_cell or not execution_tracker:
            return 0.5
        
        cell_id = exchange.get('cell_id')
        if not cell_id:
            return 0.3  # Non-cell exchanges get lower dependency score
        
        try:
            dep_chain = execution_tracker.get_dependency_chain(current_cell)
            if cell_id in dep_chain:
                return 1.0
            elif cell_id == current_cell:
                return 0.8
            else:
                return 0.2
        except:
            return 0.5
    
    def calculate_user_importance_score(self, exchange: Dict[str, Any],
                                      execution_tracker=None) -> float:
        """Calculate user-defined importance score."""
        if not execution_tracker:
            return 0.5
        
        cell_id = exchange.get('cell_id')
        if not cell_id:
            return 0.5
        
        try:
            if cell_id in execution_tracker.pinned_cells:
                return 1.0
            elif cell_id in execution_tracker.ignored_cells:
                return 0.0
            else:
                return 0.5
        except:
            return 0.5
    
    def calculate_error_resolution_score(self, exchange: Dict[str, Any]) -> float:
        """Calculate error resolution score."""
        output = exchange.get('output', '').lower()
        input_text = exchange.get('input', '').lower()
        
        # Check for error-related content
        error_keywords = ['error', 'exception', 'traceback', 'failed', 'fix', 'debug']
        solution_keywords = ['solution', 'resolved', 'fixed', 'corrected', 'works']
        
        has_error = any(keyword in input_text or keyword in output for keyword in error_keywords)
        has_solution = any(keyword in output for keyword in solution_keywords)
        
        if has_error and has_solution:
            return 0.8  # Error with solution
        elif has_error:
            return 0.6  # Error without clear solution
        elif has_solution:
            return 0.7  # Solution content
        else:
            return 0.5  # Neutral
    
    def calculate_content_quality_score(self, exchange: Dict[str, Any]) -> float:
        """Calculate content quality score."""
        input_text = exchange.get('input', '')
        output_text = exchange.get('output', '')
        
        # Length-based quality (substantial content is better)
        total_length = len(input_text) + len(output_text)
        length_score = min(1.0, total_length / 500)  # Normalize to 500 chars
        
        # Code content bonus
        code_indicators = ['def ', 'class ', 'import ', 'for ', 'if ', 'while ']
        has_code = any(indicator in input_text or indicator in output_text 
                      for indicator in code_indicators)
        
        code_bonus = 0.2 if has_code else 0.0
        
        return min(1.0, length_score + code_bonus)
    
    def summarize_exchange(self, exchange: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summarized version of an exchange."""
        input_text = exchange.get('input', '')
        output_text = exchange.get('output', '')
        
        # Simple summarization - take first and last sentences
        def summarize_text(text: str, max_length: int = 100) -> str:
            if len(text) <= max_length:
                return text
            
            sentences = text.split('. ')
            if len(sentences) <= 2:
                return text[:max_length] + '...'
            
            first_sentence = sentences[0]
            last_sentence = sentences[-1]
            
            summary = f"{first_sentence}... {last_sentence}"
            if len(summary) > max_length:
                summary = text[:max_length] + '...'
            
            return summary
        
        summarized = exchange.copy()
        summarized['input'] = summarize_text(input_text)
        summarized['output'] = summarize_text(output_text)
        summarized['is_summary'] = True
        
        return summarized
    
    def estimate_tokens(self, exchange: Dict[str, Any]) -> int:
        """Estimate token count for an exchange."""
        text = exchange.get('input', '') + exchange.get('output', '')
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_pruning_analysis(self, original_history: List[Dict[str, Any]], 
                           pruned_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis of pruning results."""
        original_count = len(original_history)
        pruned_count = len(pruned_history)
        removed_count = original_count - pruned_count
        
        # Calculate token savings
        original_tokens = sum(self.estimate_tokens(ex) for ex in original_history)
        pruned_tokens = sum(self.estimate_tokens(ex) for ex in pruned_history)
        
        # Identify what was removed
        pruned_ids = {id(ex) for ex in pruned_history}
        removed_exchanges = [ex for ex in original_history if id(ex) not in pruned_ids]
        
        return {
            'original_count': original_count,
            'pruned_count': pruned_count,
            'removed_count': removed_count,
            'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0,
            'original_tokens': original_tokens,
            'pruned_tokens': pruned_tokens,
            'token_savings': original_tokens - pruned_tokens,
            'token_savings_percentage': ((original_tokens - pruned_tokens) / original_tokens * 100) if original_tokens > 0 else 0,
            'removed_exchanges': removed_exchanges[:5],  # Sample of removed exchanges
            'avg_relevance_score': sum(ex.get('relevance_score', 0) for ex in pruned_history) / len(pruned_history) if pruned_history else 0
        }
