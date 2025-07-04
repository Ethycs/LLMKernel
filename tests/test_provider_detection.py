#!/usr/bin/env python3
"""
Unit tests for provider detection logic.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_kernel.magic_commands.multimodal_native_pdf import NativePDFMagics


class TestProviderDetection(unittest.TestCase):
    """Test provider detection from model names."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock kernel and shell
        class MockKernel:
            pass
        class MockShell:
            pass
        
        self.magic = NativePDFMagics(MockShell(), MockKernel())
    
    def test_openai_detection(self):
        """Test OpenAI model detection."""
        openai_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
            "GPT-4",  # Test case insensitivity
        ]
        
        for model in openai_models:
            with self.subTest(model=model):
                result = self.magic._get_provider_from_model(model)
                self.assertEqual(result, 'openai', f"Failed to detect OpenAI for {model}")
    
    def test_claude_detection(self):
        """Test Claude model detection."""
        claude_models = [
            "claude-3-opus",
            "claude-3-5-sonnet",
            "claude-3-haiku",
            "Claude-3-Opus",  # Test case insensitivity
        ]
        
        for model in claude_models:
            with self.subTest(model=model):
                result = self.magic._get_provider_from_model(model)
                self.assertEqual(result, 'anthropic', f"Failed to detect Anthropic for {model}")
    
    def test_gemini_detection(self):
        """Test Gemini model detection."""
        gemini_models = [
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "Gemini-Pro",  # Test case insensitivity
            "bison",  # PaLM model
        ]
        
        for model in gemini_models:
            with self.subTest(model=model):
                result = self.magic._get_provider_from_model(model)
                self.assertEqual(result, 'gemini', f"Failed to detect Gemini for {model}")
    
    def test_unknown_detection(self):
        """Test unknown model detection."""
        unknown_models = [
            "llama3",
            "mistral",
            "random-model",
            "",
            None,
        ]
        
        for model in unknown_models:
            with self.subTest(model=model):
                result = self.magic._get_provider_from_model(model)
                self.assertEqual(result, 'unknown', f"Failed to return unknown for {model}")


class TestFileUploadManager(unittest.TestCase):
    """Test file upload manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from llm_kernel.file_upload_manager import FileUploadManager
        
        # Create a mock logger
        class MockLogger:
            def info(self, msg): pass
            def error(self, msg): pass
            def warning(self, msg): pass
            def debug(self, msg): pass
        
        self.manager = FileUploadManager(logger=MockLogger())
    
    def test_mime_type_detection(self):
        """Test MIME type detection."""
        test_cases = [
            ("test.pdf", "application/pdf"),
            ("test.png", "image/png"),
            ("test.jpg", "image/jpeg"),
            ("test.txt", "text/plain"),
        ]
        
        for filename, expected in test_cases:
            with self.subTest(filename=filename):
                # Create a Path object
                from pathlib import Path
                path = Path(filename)
                result = self.manager._get_mime_type(path)
                self.assertEqual(result, expected, f"Wrong MIME type for {filename}")


if __name__ == '__main__':
    unittest.main()