"""
OpenAI File Handler - Simplified approach for PDF support

Since the Assistants API is deprecated and the new Responses API requires
vector stores, we'll use a simpler approach:
1. For vision models (GPT-4o, etc.) - convert PDFs to images
2. For text extraction - use PDF text extraction
"""

import os
import base64
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class OpenAIFileHandler:
    """Handles file processing for OpenAI models."""
    
    def __init__(self, logger=None):
        self.log = logger
        
    def process_pdf_for_vision(self, pdf_path: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Convert PDF to images for vision models like GPT-4o."""
        if not HAS_PYMUPDF:
            if self.log:
                self.log.warning("PyMuPDF not available, cannot convert PDF to images")
            return []
        
        try:
            import fitz
            pdf_document = fitz.open(pdf_path)
            
            images = []
            for page_num in range(min(len(pdf_document), max_pages)):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for clarity
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')
                
                images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })
                
            pdf_document.close()
            
            if self.log:
                self.log.info(f"Converted {len(images)} pages from PDF to images")
                
            return images
            
        except Exception as e:
            if self.log:
                self.log.error(f"Error converting PDF to images: {e}")
            return []
    
    def extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF for text-based processing."""
        if HAS_PYPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                    
                    if self.log:
                        self.log.info(f"Extracted {len(text)} characters from PDF")
                    
                    return text
                    
            except Exception as e:
                if self.log:
                    self.log.error(f"Error extracting PDF text with PyPDF2: {e}")
        
        if HAS_PYMUPDF:
            try:
                import fitz
                pdf_document = fitz.open(pdf_path)
                text = ""
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n\n"
                
                pdf_document.close()
                
                if self.log:
                    self.log.info(f"Extracted {len(text)} characters from PDF with PyMuPDF")
                
                return text
                
            except Exception as e:
                if self.log:
                    self.log.error(f"Error extracting PDF text with PyMuPDF: {e}")
        
        return None
    
    def format_pdf_for_openai(self, pdf_path: str, model: str = "gpt-4o") -> List[Dict[str, Any]]:
        """Format PDF for OpenAI model based on capabilities."""
        pdf_path = Path(pdf_path)
        
        # For vision models, convert to images
        if "gpt-4o" in model or "gpt-4-turbo" in model:
            images = self.process_pdf_for_vision(str(pdf_path))
            if images:
                return [
                    {
                        "type": "text",
                        "text": f"I've uploaded a PDF document: {pdf_path.name}. Please analyze the following pages:"
                    }
                ] + images
        
        # For non-vision models, extract text
        text = self.extract_pdf_text(str(pdf_path))
        if text:
            # Truncate if too long
            max_chars = 50000  # Reasonable limit
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Document truncated due to length...]"
            
            return [
                {
                    "type": "text",
                    "text": f"PDF Document: {pdf_path.name}\n\nContent:\n{text}"
                }
            ]
        
        # Fallback
        return [
            {
                "type": "text",
                "text": f"Unable to process PDF: {pdf_path.name}"
            }
        ]