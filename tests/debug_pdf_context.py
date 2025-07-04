#!/usr/bin/env python3
"""
Debug script to check if PDF files are properly added to conversation context.
"""

import json

# Simulate the conversation history after PDF upload
conversation_history = []

# This is what gets added by %llm_pdf_native
pdf_message = {
    "role": "user",
    "content": [
        {
            "type": "file",
            "file": {
                "file_id": "file-RVdkEWKuWzULAbZMhyeqgU",
                "filename": "the_measure_of_apocalypse.pdf"
            }
        },
        {
            "type": "text", 
            "text": "[Uploaded PDF: the_measure_of_apocalypse.pdf]"
        }
    ]
}

conversation_history.append(pdf_message)

# Add a query
query_message = {
    "role": "user",
    "content": "What is this document about?"
}
conversation_history.append(query_message)

print("Conversation History:")
print(json.dumps(conversation_history, indent=2))

# Test file ID extraction (same logic as in llm_integration.py)
def extract_file_ids(messages):
    file_ids = []
    
    for msg in messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'file' and 'file' in item:
                    file_info = item['file']
                    if 'file_id' in file_info:
                        file_ids.append(file_info['file_id'])
    
    return file_ids

file_ids = extract_file_ids(conversation_history)
print(f"\nExtracted file IDs: {file_ids}")

# Check if we would trigger Assistant API
model = "gpt-4o"
if 'gpt' in model.lower() and file_ids:
    print("\n✅ Should trigger Assistant API")
    print(f"   Model: {model}")
    print(f"   Files: {file_ids}")
else:
    print("\n❌ Would NOT trigger Assistant API")