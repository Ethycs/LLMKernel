"""
LLM Kernel Entry Point

This module provides the entry point for running the LLM kernel.
"""

import sys
from ipykernel.kernelapp import IPKernelApp
from .kernel import LLMKernel


class LLMKernelApp(IPKernelApp):
    """Application for launching the LLM kernel."""
    
    kernel_class = LLMKernel
    
    def initialize(self, argv=None):
        super().initialize(argv)
        # Additional initialization if needed


def main():
    """Main entry point for the LLM kernel."""
    app = LLMKernelApp.instance()
    app.initialize()
    app.start()


if __name__ == '__main__':
    main()
