#!/usr/bin/env python3
"""Test magic command output in chat mode."""

# This script should be run as cells in a Jupyter notebook with LLM Kernel

print("Test 1: Basic print before chat mode")
print("This should display normally")

print("\n" + "="*50 + "\n")

print("Test 2: Enable chat mode")
print("Run: %llm_chat on")
print("Expected: Should see chat mode enabled messages")

print("\n" + "="*50 + "\n")

print("Test 3: Test print in chat mode")
print("Run: print('Hello from Python')")
print("Expected: Should see 'Hello from Python' as plain text")

print("\n" + "="*50 + "\n")

print("Test 4: Test magic command in chat mode")
print("Run: %llm_status")
print("Expected: Should see kernel status as plain text")

print("\n" + "="*50 + "\n")

print("Test 5: Toggle chat mode")
print("Run: %llm_chat")
print("Expected: Should see 'Chat mode: OFF' messages")

print("\n" + "="*50 + "\n")

print("If any of these show '<IPython.core.display.HTML object>', there's a display hook issue")