"""
Multimodal content support for LLM Kernel.

Handles images, PDFs, clipboard content, and other media types
for vision-capable LLMs.
"""

import base64
import io
import os
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import requests

try:
    import fitz  # PyMuPDF for PDF handling
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

try:
    from .clipboard_utils import get_clipboard_files
    HAS_FILE_CLIPBOARD = True
except ImportError:
    HAS_FILE_CLIPBOARD = False


class MultimodalContent:
    """Handles multimodal content for LLM interactions."""
    
    def __init__(self, kernel_instance=None):
        self.kernel = kernel_instance
        self.log = kernel_instance.log if kernel_instance else None
        
        # Track multimodal content in cells
        self._cell_media = {}  # cell_id -> list of media items
        
    def encode_image(self, image_path: Union[str, Path, Image.Image]) -> Dict[str, str]:
        """
        Encode an image to base64 for LLM consumption.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Dict with 'type' and 'data' for LLM
        """
        if isinstance(image_path, Image.Image):
            # Already a PIL Image
            img = image_path
            mime_type = 'image/png'
        else:
            # Load from file
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Get MIME type
            mime_type = mimetypes.guess_type(str(image_path))[0]
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/png'
            
            # Open with PIL for potential preprocessing
            img = Image.open(image_path)
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        
        # Resize if too large (max 2048x2048 for most LLMs)
        max_size = 2048
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if self.log:
                self.log.info(f"Resized image to {img.size}")
        
        # Convert to base64
        buffer = io.BytesIO()
        img_format = 'PNG' if mime_type == 'image/png' else 'JPEG'
        img.save(buffer, format=img_format, quality=85 if img_format == 'JPEG' else None)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'type': 'image',
            'mime_type': mime_type,
            'data': img_base64,
            'size': img.size
        }
    
    def encode_pdf(self, pdf_path: Union[str, Path], 
                   pages: Optional[List[int]] = None,
                   as_images: bool = True) -> List[Dict[str, str]]:
        """
        Encode a PDF for LLM consumption.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to include (None = all)
            as_images: Convert pages to images (for vision LLMs)
            
        Returns:
            List of content items (text or images)
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz) required for PDF support. Install with: pip install pymupdf")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        content_items = []
        doc = fitz.open(pdf_path)
        
        # Determine pages to process
        if pages is None:
            pages = range(len(doc))
        
        for page_num in pages:
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            if as_images:
                # Render page as image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for clarity
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Encode as image
                img_dict = self.encode_image(img)
                img_dict['source'] = f"{pdf_path.name} (page {page_num + 1})"
                content_items.append(img_dict)
            else:
                # Extract text
                text = page.get_text()
                if text.strip():
                    content_items.append({
                        'type': 'text',
                        'data': text,
                        'source': f"{pdf_path.name} (page {page_num + 1})"
                    })
        
        doc.close()
        return content_items
    
    def get_clipboard_image(self) -> Optional[Dict[str, str]]:
        """
        Get image from clipboard if available.
        
        Returns:
            Encoded image dict or None
        """
        # First try to get image files from clipboard
        files = self.get_clipboard_files()
        if files:
            # Look for image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
            for file_info in files:
                if file_info.get('extension') in image_extensions and 'data' in file_info:
                    # Decode the base64 image to get dimensions
                    try:
                        img_bytes = base64.b64decode(file_info['data'])
                        img = Image.open(io.BytesIO(img_bytes))
                        return {
                            'type': 'image',
                            'mime_type': mimetypes.guess_type(file_info['filename'])[0] or 'image/png',
                            'data': file_info['data'],
                            'size': img.size,
                            'source': 'clipboard_file',
                            'filename': file_info['filename']
                        }
                    except Exception:
                        pass
        
        # Fallback to ImageGrab for screenshots/copied images
        try:
            from PIL import ImageGrab
            
            # Try to grab image from clipboard
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                return self.encode_image(img)
            
        except Exception as e:
            if self.log:
                self.log.debug(f"No image in clipboard: {e}")
        
        return None
    
    def get_clipboard_files(self) -> Optional[List[Dict[str, Any]]]:
        """Get files from clipboard (when copied with Ctrl+C)."""
        if HAS_FILE_CLIPBOARD:
            try:
                files = get_clipboard_files()
                if files:
                    return files
            except Exception as e:
                if self.log:
                    self.log.debug(f"Error getting clipboard files: {e}")
        
        return None
    
    def get_clipboard_pdf(self) -> Optional[Dict[str, Any]]:
        """Get PDF from clipboard if available (file path or file content)."""
        # First try to get actual files from clipboard
        files = self.get_clipboard_files()
        if files:
            # Look for PDFs in the file list
            for file_info in files:
                if file_info.get('extension') == '.pdf' and 'data' in file_info:
                    return {
                        'type': 'pdf',
                        'filename': file_info['filename'],
                        'data': file_info['data'],
                        'size': file_info['size'],
                        'source': 'clipboard_file',
                        'path': file_info['path']
                    }
        
        # Fallback to checking if clipboard contains a file path
        if HAS_CLIPBOARD:
            try:
                # Get text from clipboard (might be a file path)
                text = pyperclip.paste()
                if text and text.strip().lower().endswith('.pdf'):
                    # It's a PDF file path
                    pdf_path = Path(text.strip().strip('"'))  # Remove quotes if present
                    if pdf_path.exists():
                        # Read and encode the PDF
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        return {
                            'type': 'pdf',
                            'filename': pdf_path.name,
                            'data': base64.b64encode(pdf_bytes).decode('utf-8'),
                            'size': len(pdf_bytes),
                            'source': 'clipboard_path',
                            'path': str(pdf_path)
                        }
                
            except Exception as e:
                if self.log:
                    self.log.debug(f"Error checking clipboard for PDF path: {e}")
        
        return None
    
    def get_clipboard_text(self) -> Optional[str]:
        """Get text from clipboard if available."""
        if not HAS_CLIPBOARD:
            return None
            
        try:
            return pyperclip.paste()
        except Exception:
            return None
    
    def download_image(self, url: str) -> Dict[str, str]:
        """Download and encode an image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Load image from response
            img = Image.open(io.BytesIO(response.content))
            img_dict = self.encode_image(img)
            img_dict['source'] = url
            return img_dict
            
        except Exception as e:
            raise ValueError(f"Failed to download image from {url}: {e}")
    
    def format_for_llm(self, content_items: List[Dict[str, Any]], 
                      model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Format multimodal content for specific LLM format.
        
        Args:
            content_items: List of content items (text, images, etc.)
            model: Model name to format for
            
        Returns:
            Formatted messages for LLM
        """
        # Determine if model supports vision
        vision_models = [
            'gpt-4-vision', 'gpt-4o', 'gpt-4o-mini',
            'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
            'gemini-pro-vision', 'gemini-1.5-pro',
            'llava', 'bakllava'
        ]
        
        supports_vision = any(vm in str(model).lower() for vm in vision_models) if model else True
        
        if not supports_vision:
            # Extract only text content
            text_items = []
            for item in content_items:
                if item['type'] == 'text':
                    text_items.append(item['data'])
                elif item['type'] == 'image':
                    text_items.append(f"[Image: {item.get('source', 'embedded')}]")
            
            return [{'role': 'user', 'content': '\n\n'.join(text_items)}]
        
        # Format for vision-capable models
        formatted_content = []
        
        for item in content_items:
            if item['type'] == 'text':
                formatted_content.append({
                    'type': 'text',
                    'text': item['data']
                })
            elif item['type'] == 'image':
                # Different formats for different providers
                if 'gpt' in str(model).lower():
                    # OpenAI format
                    formatted_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:{item['mime_type']};base64,{item['data']}"
                        }
                    })
                elif 'claude' in str(model).lower():
                    # Anthropic format
                    formatted_content.append({
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': item['mime_type'],
                            'data': item['data']
                        }
                    })
                else:
                    # Generic format
                    formatted_content.append({
                        'type': 'image',
                        'data': f"data:{item['mime_type']};base64,{item['data']}"
                    })
        
        return [{'role': 'user', 'content': formatted_content}]
    
    def add_to_cell(self, cell_id: str, content_item: Dict[str, Any]):
        """Add multimodal content to a cell's context."""
        if cell_id not in self._cell_media:
            self._cell_media[cell_id] = []
        self._cell_media[cell_id].append(content_item)
    
    def get_cell_media(self, cell_id: str) -> List[Dict[str, Any]]:
        """Get all media items for a cell."""
        return self._cell_media.get(cell_id, [])