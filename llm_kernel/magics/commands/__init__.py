"""
Magic Commands for LLM Kernel

This package contains all magic command implementations,
organized by functionality.
"""

from llm_kernel.magics.commands.base import BaseMagics
from llm_kernel.magics.commands.context import ContextMagics
from llm_kernel.magics.commands.mcp import MCPMagics
from llm_kernel.magics.commands.reranking import RerankingMagics
from llm_kernel.magics.commands.config import ConfigMagics

__all__ = [
    'BaseMagics',
    'ContextMagics', 
    'MCPMagics',
    'RerankingMagics',
    'ConfigMagics'
]