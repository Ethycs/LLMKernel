#!/usr/bin/env python3
"""
Test script for OpenAI Assistant PDF upload functionality
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from llm_kernel.openai_assistant_integration import OpenAIAssistantIntegration
from llm_kernel.file_upload_manager import FileUploadManager

def test_pdf_upload():
    """Test PDF upload and query with OpenAI Assistant API."""
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not set")
        return
    
    # Create a simple test PDF
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a test PDF
        pdf_path = "test_document.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "Test Document for LLM Kernel")
        c.drawString(100, 700, "This is a test PDF file.")
        c.drawString(100, 650, "The answer to the test question is: 42")
        c.drawString(100, 600, "This document contains important information.")
        c.save()
        print(f"Created test PDF: {pdf_path}")
        
    except ImportError:
        print("reportlab not installed, using existing PDF")
        pdf_path = input("Enter path to a PDF file: ").strip()
        if not Path(pdf_path).exists():
            print(f"File not found: {pdf_path}")
            return
    
    # Initialize components
    print("\nInitializing components...")
    
    # Create a simple logger
    class SimpleLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): pass
        def warning(self, msg): print(f"[WARN] {msg}")
    
    logger = SimpleLogger()
    
    # Initialize file upload manager
    upload_manager = FileUploadManager(logger=logger)
    
    # Set OpenAI client
    import openai
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    upload_manager.set_provider_client('openai', client)
    
    # Upload the PDF
    print(f"\nUploading PDF: {pdf_path}")
    upload_info = upload_manager.upload_file(
        pdf_path,
        purpose="assistants",
        provider="openai"
    )
    
    if not upload_info or 'file_id' not in upload_info:
        print("Failed to upload PDF")
        return
    
    file_id = upload_info['file_id']
    print(f"Successfully uploaded PDF with file_id: {file_id}")
    
    # Initialize assistant integration
    assistant = OpenAIAssistantIntegration(logger=logger)
    
    # Query with the uploaded file
    print("\nQuerying assistant with uploaded PDF...")
    query = "What is the answer to the test question mentioned in this document?"
    
    result = assistant.query_with_files(
        query=query,
        file_ids=[file_id],
        model="gpt-4o"
    )
    
    if result:
        print(f"\nAssistant response:\n{result}")
    else:
        print("Failed to get response from assistant")
    
    # Clean up
    if Path("test_document.pdf").exists() and "test_document" in pdf_path:
        os.remove("test_document.pdf")
        print("\nCleaned up test PDF")

if __name__ == "__main__":
    print("Testing OpenAI Assistant PDF functionality...")
    test_pdf_upload()