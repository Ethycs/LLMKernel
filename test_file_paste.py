"""Test file paste functionality with Ctrl+C/Ctrl+V support."""

print("""
Enhanced File Paste Support
===========================

The %llm_paste command now supports multiple ways to paste files:

1. **Copy Files with Ctrl+C** (NEW!)
   - Select files in File Explorer/Finder
   - Press Ctrl+C (or Cmd+C on Mac)
   - Run %llm_paste in Jupyter
   - Files are automatically uploaded!

2. **Copy File Paths**
   - Right-click → Copy as path
   - Run %llm_paste

3. **Copy Images**
   - Copy any image (screenshot, from web, etc.)
   - Run %llm_paste

Supported file types when using Ctrl+C:
- PDFs (.pdf)
- Images (.png, .jpg, .jpeg, .gif, .bmp)

Platform Requirements:
- Windows: pip install pywin32
- macOS: Works out of the box
- Linux: sudo apt-get install xclip

Test Instructions:
==================

1. Copy a PDF file with Ctrl+C:
   - Open File Explorer
   - Select a PDF file
   - Press Ctrl+C
   - Run: %llm_paste
   
2. Check what's in clipboard:
   %llm_paste --show

3. Ask about the file:
   What's in this PDF?

The file is uploaded to the conversation and persists across cells!

Note: This feature detects actual file data in the clipboard,
not just file paths, making it work exactly like modern chat apps.
""")