"""Test native PDF upload functionality in LLM Kernel."""

print("""
Native PDF Upload Test
======================

The LLM Kernel now supports two ways to handle PDFs:

1. Image Conversion (Original Method - %llm_pdf):
   - Converts each page to an image
   - Works with any vision model
   - Good for visual elements, diagrams, layouts
   
2. Native Upload (New Method - %llm_pdf_native):
   - Uploads the PDF file directly
   - Supported by OpenAI GPT-4.1+, Claude, etc.
   - Better for text extraction and full document understanding
   - More efficient (no conversion needed)

Test Instructions:
==================

1. Find a PDF file to test with

2. Try the image conversion method:
   %llm_pdf test.pdf --pages 1,2
   What's on the first two pages?

3. Try the native upload method:
   %llm_pdf_native test.pdf
   Summarize this entire document

4. List your uploaded files:
   %llm_files_list
   
5. Clear files when done:
   %llm_files_clear

Note: The native method sends the entire PDF file to the API,
which may be more cost-effective and accurate for text-heavy documents.
""")