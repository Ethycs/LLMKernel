#!/usr/bin/env python3
"""
Test to reproduce the chat mode issue where magic commands 
outputs are shown as HTML objects.
"""

import subprocess
import sys

# Create a test notebook
test_notebook = '''
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Enable chat mode\\n",
    "%llm_chat on"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Regular Python code\\n",
    "print('Hello from Python')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Magic command\\n",
    "%llm_models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Another magic command\\n",
    "%llm_context"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "LLM Kernel",
   "language": "python",
   "name": "llm_kernel"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

# Write test notebook
with open('test_chat_mode.ipynb', 'w') as f:
    f.write(test_notebook)

print("Test notebook created: test_chat_mode.ipynb")
print("Run it with: jupyter notebook test_chat_mode.ipynb")
print("\nExpected behavior:")
print("- Cell 1: Should show chat mode enabled message")
print("- Cell 2: Should print 'Hello from Python'")
print("- Cell 3: Should list available models")
print("- Cell 4: Should show context information")
print("\nIssue: If cells 2-4 show '<IPython.core.display.HTML object>' then we have the bug")