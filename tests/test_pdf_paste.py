"""Test PDF paste functionality in LLM Kernel."""

print("""
PDF Paste Test Instructions
===========================

The %llm_paste command now supports PDFs from clipboard!

How it works:
1. Copy a PDF file path to your clipboard
   - In Windows Explorer: Right-click PDF → Copy as path
   - In Terminal: Copy the full path like "/home/user/document.pdf"
   - On Mac: Right-click PDF → Option+Copy → Copy as Pathname

2. Use %llm_paste to paste the PDF:
   %llm_paste
   ✅ Pasted PDF 'document.pdf' (2.3 MB) - added to conversation context

3. Ask questions about it in any cell:
   What's this PDF about?
   Summarize the main points
   Extract all the tables

The PDF is uploaded to the conversation context and persists across cells!

You can also check what's in your clipboard first:
%llm_paste --show

Note: This works by detecting that the clipboard contains a path ending in .pdf
and automatically reading and uploading that file.
""")