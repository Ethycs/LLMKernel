#!/usr/bin/env python3
"""
Quick test runner for PDF upload functionality.
Run this to verify all providers are working correctly.
"""

import os
import sys
from pathlib import Path

# Default test PDF
DEFAULT_PDF = r"F:\Keytone\OneDrive\LaTex\Tex\AI_Research\dense_humans\the_measure_of_apocalypse.pdf"

def main():
    """Run PDF upload tests."""
    
    # Check for custom PDF path
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    else:
        test_pdf = DEFAULT_PDF
    
    # Verify PDF exists
    if not Path(test_pdf).exists():
        print(f"Error: PDF not found: {test_pdf}")
        print(f"\nUsage: {sys.argv[0]} [path/to/test.pdf]")
        return 1
    
    print("="*60)
    print("Running PDF Upload Tests")
    print(f"Test PDF: {test_pdf}")
    print("="*60)
    
    # Run the main test suite
    print("\n1. Running provider tests...")
    os.system(f"python {Path(__file__).parent}/test_multimodal_pdf_upload.py")
    
    # Run the integration test
    print("\n2. Running integration test...")
    os.system(f"python {Path(__file__).parent}/test_pdf_notebook_integration.py")
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("\nTo test in a notebook, open:")
    print(f"  {Path(__file__).parent}/test_pdf_providers.ipynb")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())