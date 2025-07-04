#!/usr/bin/env python
"""Check multimodal dependencies for LLM Kernel."""

print("Checking multimodal dependencies...\n")

# Check PIL/Pillow
try:
    from PIL import Image
    print("✓ PIL.Image imported successfully")
    import PIL
    print(f"  Version: {PIL.__version__}")
except ImportError as e:
    print(f"✗ PIL.Image import failed: {e}")

# Check pyperclip
try:
    import pyperclip
    print("✓ pyperclip imported successfully")
    # Try to get clipboard (might fail in some environments)
    try:
        content = pyperclip.paste()
        print(f"  Clipboard access works (content length: {len(content)})")
    except Exception as e:
        print(f"  Clipboard access failed: {e}")
except ImportError as e:
    print(f"✗ pyperclip import failed: {e}")

# Check PyMuPDF
try:
    import fitz
    print("✓ fitz (PyMuPDF) imported successfully")
    print(f"  Version: {fitz.version}")
except ImportError as e:
    print(f"✗ fitz (PyMuPDF) import failed: {e}")

# Check if running in Jupyter
print("\nEnvironment info:")
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# Check if we're in a pixi environment
import os
if 'PIXI_ENVIRONMENT_NAME' in os.environ:
    print(f"Pixi environment: {os.environ['PIXI_ENVIRONMENT_NAME']}")

# Try importing the multimodal module
print("\nChecking LLM Kernel multimodal module...")
try:
    from llm_kernel.multimodal import MultimodalContent
    print("✓ MultimodalContent imported successfully")
except ImportError as e:
    print(f"✗ MultimodalContent import failed: {e}")

try:
    from llm_kernel.magic_commands.multimodal import MultimodalMagics
    print("✓ MultimodalMagics imported successfully")
except ImportError as e:
    print(f"✗ MultimodalMagics import failed: {e}")