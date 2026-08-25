"""
LLM Jupyter Kernel with Context Management

A Jupyter kernel that uses LiteLLM for multi-provider LLM access with intelligent
context window management based on cell execution order and dependencies.
"""

__version__ = "0.1.0"
__author__ = "LLM Kernel Team"

# Embeddable facade — the public library entry point.
# See docs/06 - Roadmaps/PLAN-kernel-facade.md and
# docs/04 - Reference/kernel/_wiring-audit.md.
from llm_kernel.kernel_facade import Kernel, KernelStartError

__all__ = ["Kernel", "KernelStartError"]
