"""
LLM Jupyter Kernel with Context Management

A Jupyter kernel that uses LiteLLM for multi-provider LLM access with intelligent
context window management based on cell execution order and dependencies.
"""

from llm_kernel.wire import versions as _versions

__version__ = _versions.PACKAGE_VERSION
__author__ = "LLM Kernel Team"

# Embeddable facade — the public library entry point.
# See docs/06 - Roadmaps/PLAN-kernel-facade.md and
# docs/04 - Reference/kernel/_wiring-audit.md.
from llm_kernel.kernel_facade import Kernel, KernelStartError

__all__ = ["Kernel", "KernelStartError"]
