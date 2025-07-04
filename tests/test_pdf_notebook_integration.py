#!/usr/bin/env python3
"""
Integration test for PDF upload in notebook context.
This simulates how the magic commands would work.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_PDF_PATH = r"F:\Keytone\OneDrive\LaTex\Tex\AI_Research\dense_humans\the_measure_of_apocalypse.pdf"


def test_pdf_upload_simulation():
    """Simulate PDF upload as it would work in a notebook."""
    
    print("Testing PDF Upload Integration")
    print("="*60)
    
    # Check if test PDF exists
    if not Path(TEST_PDF_PATH).exists():
        print(f"ERROR: Test PDF not found: {TEST_PDF_PATH}")
        return
    
    # Simulate kernel initialization
    print("\n1. Simulating kernel initialization...")
    
    # Create a mock kernel object
    class MockKernel:
        def __init__(self):
            self.active_model = "gpt-4o"  # Change to test different providers
            self.conversation_history = []
            self.log = self._create_logger()
            self._uploaded_files = []
            
        def _create_logger(self):
            class Logger:
                def info(self, msg): print(f"[KERNEL] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
                def warning(self, msg): print(f"[WARN] {msg}")
                def debug(self, msg): pass
            return Logger()
    
    kernel = MockKernel()
    
    # Test different models
    models_to_test = [
        ("gpt-4o", "OpenAI"),
        ("claude-3-5-sonnet", "Claude"),
        ("gemini-2.0-flash", "Gemini")
    ]
    
    for model, provider in models_to_test:
        print(f"\n2. Testing with {provider} ({model})...")
        kernel.active_model = model
        kernel.conversation_history = []
        
        # Simulate the %llm_pdf_native command
        print(f"   Simulating: %llm_pdf_native {Path(TEST_PDF_PATH).name}")
        
        try:
            # This simulates what the magic command does
            from llm_kernel.magic_commands.multimodal_native_pdf import NativePDFMagics
            
            # Create magic instance
            class MockShell:
                pass
            
            magic = NativePDFMagics(MockShell(), kernel)
            
            # Simulate the command execution
            magic.llm_pdf_native(TEST_PDF_PATH)
            
            # Check if PDF was added to conversation
            if kernel.conversation_history:
                print(f"   ✅ PDF added to conversation history")
                
                # Print the message structure
                msg = kernel.conversation_history[-1]
                print(f"   Message structure: {msg['role']}")
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item.get('type') == 'file':
                            print(f"   - File reference added: {item}")
                        elif item.get('type') == 'document':
                            print(f"   - Document reference added: {item}")
                        elif item.get('type') == 'text':
                            print(f"   - Text: {item.get('text', '')}")
            else:
                print(f"   ❌ No PDF added to conversation")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Integration test complete!")


def test_provider_detection():
    """Test the provider detection logic."""
    print("\nTesting Provider Detection")
    print("="*60)
    
    from llm_kernel.magic_commands.multimodal_native_pdf import NativePDFMagics
    
    # Create a mock instance
    class MockKernel:
        pass
    class MockShell:
        pass
    
    magic = NativePDFMagics(MockShell(), MockKernel())
    
    test_cases = [
        ("gpt-4o", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("claude-3-opus", "anthropic"),
        ("claude-3-5-sonnet", "anthropic"),
        ("gemini-pro", "gemini"),
        ("gemini-2.0-flash", "gemini"),
        ("llama3", "unknown"),
        ("", "unknown")
    ]
    
    for model, expected in test_cases:
        result = magic._get_provider_from_model(model)
        status = "✅" if result == expected else "❌"
        print(f"{status} {model} -> {result} (expected: {expected})")


if __name__ == "__main__":
    test_pdf_upload_simulation()
    test_provider_detection()