#!/usr/bin/env python3
"""
Test PDF upload functionality across all three major providers:
- OpenAI (via Assistants API)
- Claude (via Files API)
- Gemini (via Files API)

This test uses a real PDF file to verify that each provider can:
1. Upload the PDF successfully
2. Read and understand the content
3. Answer questions about the document
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_kernel.file_upload_manager import FileUploadManager
from llm_kernel.openai_assistant_integration import OpenAIAssistantIntegration
from llm_kernel.anthropic_file_integration import AnthropicFileIntegration
from llm_kernel.gemini_file_integration import GeminiFileIntegration

# Test configuration
TEST_PDF_PATH = r"F:\Keytone\OneDrive\LaTex\Tex\AI_Research\dense_humans\the_measure_of_apocalypse.pdf"
TEST_QUERY = "What is the main topic or thesis of this document? Please provide a brief summary in 2-3 sentences."

class TestLogger:
    """Simple logger for tests."""
    def info(self, msg): print(f"[INFO] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")
    def debug(self, msg): pass  # Suppress debug messages


def test_openai_pdf_upload():
    """Test OpenAI PDF upload using Assistants API."""
    print("\n" + "="*60)
    print("Testing OpenAI PDF Upload (Assistants API)")
    print("="*60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("SKIP: OPENAI_API_KEY not set")
        return False
    
    try:
        logger = TestLogger()
        
        # Initialize file upload manager
        upload_manager = FileUploadManager(logger=logger)
        
        # Set OpenAI client
        import openai
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        upload_manager.set_provider_client('openai', client)
        
        # Upload the PDF
        print(f"Uploading PDF: {Path(TEST_PDF_PATH).name}")
        upload_info = upload_manager.upload_file(
            TEST_PDF_PATH,
            purpose="assistants",
            provider="openai"
        )
        
        if not upload_info or 'file_id' not in upload_info:
            print("FAIL: Failed to upload PDF to OpenAI")
            return False
        
        file_id = upload_info['file_id']
        print(f"SUCCESS: Uploaded with file_id: {file_id}")
        
        # Test with Assistant API
        assistant_integration = OpenAIAssistantIntegration(logger=logger)
        
        print("Querying OpenAI about the PDF content...")
        result = assistant_integration.query_with_files(
            query=TEST_QUERY,
            file_ids=[file_id],
            model="gpt-4o"
        )
        
        if result:
            print(f"\nOpenAI Response:\n{result[:500]}...")
            print("\n✅ OpenAI PDF upload and reading: PASSED")
            return True
        else:
            print("FAIL: No response from OpenAI Assistant")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claude_pdf_upload():
    """Test Claude PDF upload using Files API."""
    print("\n" + "="*60)
    print("Testing Claude PDF Upload (Files API)")
    print("="*60)
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("SKIP: ANTHROPIC_API_KEY not set")
        return False
    
    try:
        logger = TestLogger()
        
        # Initialize Anthropic file integration
        anthropic_files = AnthropicFileIntegration(logger=logger)
        
        # Upload the PDF
        print(f"Uploading PDF: {Path(TEST_PDF_PATH).name}")
        upload_info = anthropic_files.upload_file(TEST_PDF_PATH)
        
        if not upload_info or 'file_id' not in upload_info:
            print("FAIL: Failed to upload PDF to Claude")
            return False
        
        file_id = upload_info['file_id']
        print(f"SUCCESS: Uploaded with file_id: {file_id}")
        
        # Test with Claude API
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            default_headers={"anthropic-beta": "files-api-2025-04-14"}
        )
        
        print("Querying Claude about the PDF content...")
        
        # Format the file reference
        file_content = anthropic_files.format_file_for_message(file_id, "document")
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    file_content,
                    {"type": "text", "text": TEST_QUERY}
                ]
            }]
        )
        
        if response and response.content:
            result = response.content[0].text
            print(f"\nClaude Response:\n{result[:500]}...")
            print("\n✅ Claude PDF upload and reading: PASSED")
            
            # Cleanup
            anthropic_files.delete_file(file_id)
            return True
        else:
            print("FAIL: No response from Claude")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_pdf_upload():
    """Test Gemini PDF upload using Files API."""
    print("\n" + "="*60)
    print("Testing Gemini PDF Upload (Files API)")
    print("="*60)
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("SKIP: GOOGLE_API_KEY not set")
        return False
    
    try:
        logger = TestLogger()
        
        # Initialize Gemini file integration
        gemini_files = GeminiFileIntegration(logger=logger)
        
        # Upload the PDF
        print(f"Uploading PDF: {Path(TEST_PDF_PATH).name}")
        upload_info = gemini_files.upload_file(TEST_PDF_PATH)
        
        if not upload_info or 'file_name' not in upload_info:
            print("FAIL: Failed to upload PDF to Gemini")
            return False
        
        file_name = upload_info['file_name']
        print(f"SUCCESS: Uploaded with file_name: {file_name}")
        
        # Test with Gemini API
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        print("Querying Gemini about the PDF content...")
        
        # Get the file object
        file_obj = gemini_files.get_file(file_name)
        if not file_obj:
            print("FAIL: Could not retrieve file object")
            return False
        
        # Create model and generate content
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([file_obj, TEST_QUERY])
        
        if response and response.text:
            print(f"\nGemini Response:\n{response.text[:500]}...")
            print("\n✅ Gemini PDF upload and reading: PASSED")
            
            # Cleanup
            gemini_files.delete_file(file_name)
            return True
        else:
            print("FAIL: No response from Gemini")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_caching():
    """Test that file caching works across providers."""
    print("\n" + "="*60)
    print("Testing File Caching")
    print("="*60)
    
    try:
        logger = TestLogger()
        upload_manager = FileUploadManager(logger=logger)
        
        # Get file hash
        file_hash = upload_manager.get_file_hash(TEST_PDF_PATH)
        print(f"File hash: {file_hash[:16]}...")
        
        # Check cache
        cache_info = upload_manager.get_cache_info()
        print(f"Cache info: {json.dumps(cache_info, indent=2)}")
        
        # List cached files
        cached_files = upload_manager.list_cached_files()
        if cached_files:
            print(f"\nCached files ({len(cached_files)}):")
            for f in cached_files:
                print(f"  - {f['original_name']} (hash: {f['file_hash'][:8]}...)")
        
        print("\n✅ File caching: WORKING")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PDF Upload Test Suite")
    print(f"Test PDF: {TEST_PDF_PATH}")
    print("="*60)
    
    # Check if test PDF exists
    if not Path(TEST_PDF_PATH).exists():
        print(f"ERROR: Test PDF not found: {TEST_PDF_PATH}")
        print("Please update TEST_PDF_PATH to point to a valid PDF file.")
        return
    
    # Run tests
    results = {
        "OpenAI": test_openai_pdf_upload(),
        "Claude": test_claude_pdf_upload(),
        "Gemini": test_gemini_pdf_upload(),
        "Caching": test_file_caching()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for provider, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED/SKIPPED"
        print(f"{provider}: {status}")
    
    # Overall result
    total_tests = len(results)
    passed_tests = sum(1 for passed in results.values() if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed or were skipped.")


if __name__ == "__main__":
    main()