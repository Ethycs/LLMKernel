#!/usr/bin/env python3
"""
Debug script to diagnose PDF upload issues.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_PDF = r"F:\Keytone\OneDrive\LaTex\Tex\AI_Research\dense_humans\the_measure_of_apocalypse.pdf"


def diagnose_openai_pdf():
    """Diagnose OpenAI PDF upload and reading."""
    print("="*60)
    print("Diagnosing OpenAI PDF Upload")
    print("="*60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    # Step 1: Test file upload
    print("\n1. Testing file upload...")
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        with open(TEST_PDF, 'rb') as f:
            file_obj = client.files.create(
                file=f,
                purpose="assistants"
            )
        
        print(f"✅ File uploaded successfully!")
        print(f"   File ID: {file_obj.id}")
        print(f"   Status: {file_obj.status}")
        print(f"   Bytes: {file_obj.bytes}")
        
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        return
    
    # Step 2: Create assistant
    print("\n2. Creating assistant...")
    try:
        assistant = client.beta.assistants.create(
            name="PDF Test Assistant",
            instructions="You are a helpful assistant that can read and analyze PDF documents.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o"
        )
        print(f"✅ Assistant created: {assistant.id}")
    except Exception as e:
        print(f"❌ Assistant creation failed: {e}")
        return
    
    # Step 3: Create thread
    print("\n3. Creating thread...")
    try:
        thread = client.beta.threads.create()
        print(f"✅ Thread created: {thread.id}")
    except Exception as e:
        print(f"❌ Thread creation failed: {e}")
        return
    
    # Step 4: Add message with file
    print("\n4. Adding message with file attachment...")
    try:
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="What is the main topic of this PDF? Please provide a brief summary.",
            attachments=[{
                "file_id": file_obj.id,
                "tools": [{"type": "code_interpreter"}]
            }]
        )
        print(f"✅ Message created: {message.id}")
    except Exception as e:
        print(f"❌ Message creation failed: {e}")
        return
    
    # Step 5: Run assistant
    print("\n5. Running assistant...")
    try:
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        print(f"✅ Run started: {run.id}")
        
        # Wait for completion
        import time
        while run.status in ['queued', 'in_progress']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            print(f"   Status: {run.status}")
        
        if run.status == 'completed':
            print("✅ Run completed successfully!")
        else:
            print(f"❌ Run failed with status: {run.status}")
            if hasattr(run, 'last_error'):
                print(f"   Error: {run.last_error}")
            return
            
    except Exception as e:
        print(f"❌ Run failed: {e}")
        return
    
    # Step 6: Get response
    print("\n6. Getting response...")
    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order='desc',
            limit=1
        )
        
        if messages.data:
            response = messages.data[0]
            print(f"✅ Got response!")
            print(f"\nAssistant says:")
            print("-"*40)
            for content in response.content:
                if content.type == 'text':
                    print(content.text.value)
            print("-"*40)
        else:
            print("❌ No response received")
            
    except Exception as e:
        print(f"❌ Failed to get response: {e}")
    
    # Cleanup
    print("\n7. Cleaning up...")
    try:
        client.beta.assistants.delete(assistant.id)
        client.files.delete(file_obj.id)
        print("✅ Cleanup complete")
    except:
        print("⚠️  Cleanup failed (not critical)")


def check_conversation_history():
    """Check what's in the conversation history."""
    print("\n" + "="*60)
    print("Checking Conversation History Structure")
    print("="*60)
    
    # Simulate kernel state
    conversation_history = []
    
    # Add a PDF upload message (as it would be added by %llm_pdf_native)
    pdf_message = {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "file_id": "file-ABC123",
                    "filename": "test.pdf"
                }
            },
            {
                "type": "text", 
                "text": "[Uploaded PDF: test.pdf]"
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
    
    print("\nConversation history structure:")
    print(json.dumps(conversation_history, indent=2))
    
    # Check file extraction
    print("\nExtracting file IDs...")
    file_ids = []
    for msg in conversation_history:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'file' and 'file' in item:
                    file_info = item['file']
                    if 'file_id' in file_info:
                        file_ids.append(file_info['file_id'])
                        print(f"✅ Found file_id: {file_info['file_id']}")
    
    if not file_ids:
        print("❌ No file IDs found in conversation history!")


def main():
    """Run diagnostics."""
    print("PDF Upload Diagnostic Tool")
    print("="*60)
    
    # Check PDF exists
    if not Path(TEST_PDF).exists():
        print(f"ERROR: Test PDF not found: {TEST_PDF}")
        return
    
    # Run diagnostics
    diagnose_openai_pdf()
    check_conversation_history()
    
    print("\n" + "="*60)
    print("Diagnostics complete!")
    print("\nIf the assistant can read the PDF in the diagnostic but not in the notebook,")
    print("the issue is likely in the integration layer between the kernel and the API.")


if __name__ == "__main__":
    main()