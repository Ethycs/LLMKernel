#!/usr/bin/env python3
"""
Test script to verify that magic commands work correctly in chat mode.
Run this in a Jupyter notebook with the LLM kernel.
"""

print("=== Testing Chat Mode with Magic Commands ===\n")

# Test sequence
test_sequence = """
# Cell 1: Enable chat mode
%llm_chat on

# Cell 2: Test a magic command
%llm_models

# Cell 3: Test another magic command  
%llm_status

# Cell 4: Test regular Python code
print("Hello from Python!")
x = 42
print(f"x = {x}")

# Cell 5: Test LLM query (should go to LLM)
What is the value of x?

# Cell 6: Test context magic
%llm_context

# Cell 7: Disable chat mode
%llm_chat off

# Cell 8: Test magic after disabling chat
%llm_status
"""

print("Run these commands in separate cells:")
print(test_sequence)
print("\nExpected behavior:")
print("- Cells 1, 2, 3, 6, 7, 8: Magic commands should execute normally with text output")
print("- Cell 4: Python code should execute normally with print output")  
print("- Cell 5: Should be sent to LLM and return HTML-formatted response")
print("\nIf any magic command or Python code shows '<IPython.core.display.HTML object>', there's a bug.")