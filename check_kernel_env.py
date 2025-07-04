"""Check the environment from within a Jupyter cell."""

def check_env():
    import sys
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    print("\nPython path:")
    for p in sys.path[:5]:  # First 5 paths
        print(f"  {p}")
    
    print("\nChecking imports:")
    try:
        from PIL import Image
        import PIL
        print(f"✓ PIL {PIL.__version__}")
    except ImportError as e:
        print(f"✗ PIL: {e}")
    
    try:
        import pyperclip
        print("✓ pyperclip")
    except ImportError as e:
        print(f"✗ pyperclip: {e}")
    
    try:
        import fitz
        print(f"✓ PyMuPDF {fitz.version}")
    except ImportError as e:
        print(f"✗ PyMuPDF: {e}")

print("Run this in a Jupyter cell:")
print("exec(open('check_kernel_env.py').read())")
print("check_env()")