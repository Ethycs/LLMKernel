"""
Magic Commands for LLM Kernel

This package contains all magic command implementations,
organized by functionality.
"""

from .base import BaseMagics
from .context import ContextMagics
from .mcp import MCPMagics
from .reranking import RerankingMagics
from .config import ConfigMagics

__all__ = [
    'BaseMagics',
    'ContextMagics', 
    'MCPMagics',
    'RerankingMagics',
    'ConfigMagics'
]