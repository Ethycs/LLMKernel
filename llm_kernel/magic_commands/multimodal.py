"""
Multimodal Magic Commands for LLM Kernel

Handles images, PDFs, clipboard content, and other media types.
"""

import os
import json
from pathlib import Path
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
from IPython.display import display, Image as IPImage, Markdown
from IPython.core.display import HTML

try:
    from ..multimodal import MultimodalContent
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False


@magics_class
class MultimodalMagics(Magics):
    """Magic commands for multimodal content."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
        self.multimodal = MultimodalContent(kernel_instance) if HAS_MULTIMODAL else None
    
    @line_magic
    def llm_paste(self, line):
        """Paste and include clipboard content (image or text) in the next LLM query.
        
        Usage:
            %llm_paste                    # Paste clipboard content
            %llm_paste --show             # Show what's in clipboard
            %llm_paste --as-image         # Force treat as image
            %llm_paste --as-text          # Force treat as text
        """
        if not self.multimodal:
            print("❌ Multimodal support not available. Install required packages.")
            return
        
        args = line.strip().split()
        show_only = '--show' in args
        force_image = '--as-image' in args
        force_text = '--as-text' in args
        
        # Try to get clipboard content
        img_content = None
        text_content = None
        
        if not force_text:
            img_content = self.multimodal.get_clipboard_image()
        
        if not force_image and not img_content:
            text_content = self.multimodal.get_clipboard_text()
        
        if not img_content and not text_content:
            print("❌ No content found in clipboard")
            return
        
        if show_only:
            if img_content:
                print("📋 Clipboard contains an image:")
                display(IPImage(data=img_content['data'], format='png'))
            elif text_content:
                print("📋 Clipboard contains text:")
                print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
            return
        
        # Add to current cell's multimodal content
        cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
        
        if img_content:
            self.multimodal.add_to_cell(cell_id, img_content)
            print(f"✅ Pasted image ({img_content['size'][0]}x{img_content['size'][1]}) - will be included in next LLM query")
            # Show thumbnail
            display(IPImage(data=img_content['data'], format='png', width=200))
        elif text_content:
            self.multimodal.add_to_cell(cell_id, {
                'type': 'text',
                'data': text_content,
                'source': 'clipboard'
            })
            print(f"✅ Pasted {len(text_content)} characters of text")
    
    @line_magic
    def llm_image(self, line):
        """Include an image file in the next LLM query.
        
        Usage:
            %llm_image path/to/image.png              # Include local image
            %llm_image https://example.com/image.jpg  # Include image from URL
            %llm_image --show path/to/image.png       # Preview without including
        """
        if not self.multimodal:
            print("❌ Multimodal support not available. Install required packages.")
            return
        
        args = line.strip().split()
        if not args:
            print("❌ Please provide an image path or URL")
            return
        
        show_only = '--show' in args
        if show_only:
            args.remove('--show')
        
        image_path = ' '.join(args)
        
        try:
            # Check if it's a URL
            if image_path.startswith(('http://', 'https://')):
                print(f"🌐 Downloading image from {image_path}...")
                img_content = self.multimodal.download_image(image_path)
            else:
                # Local file
                img_content = self.multimodal.encode_image(image_path)
                img_content['source'] = image_path
            
            if show_only:
                print(f"🖼️ Preview of {image_path}:")
                display(IPImage(data=img_content['data'], format='png', width=400))
                return
            
            # Add to current cell's multimodal content
            cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
            self.multimodal.add_to_cell(cell_id, img_content)
            
            print(f"✅ Added image ({img_content['size'][0]}x{img_content['size'][1]}) - will be included in next LLM query")
            # Show thumbnail
            display(IPImage(data=img_content['data'], format='png', width=200))
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
    
    @line_magic
    def llm_pdf(self, line):
        """Include PDF content in the next LLM query.
        
        Usage:
            %llm_pdf path/to/document.pdf           # Include all pages as images
            %llm_pdf --pages 1,3,5 document.pdf     # Include specific pages
            %llm_pdf --text document.pdf            # Extract text instead of images
            %llm_pdf --show document.pdf            # Preview first page
        """
        if not self.multimodal:
            print("❌ Multimodal support not available. Install required packages.")
            return
        
        args = line.strip().split()
        if not args:
            print("❌ Please provide a PDF path")
            return
        
        # Parse arguments
        pages = None
        as_text = False
        show_only = False
        pdf_path = None
        
        i = 0
        while i < len(args):
            if args[i] == '--pages' and i + 1 < len(args):
                pages = [int(p) - 1 for p in args[i + 1].split(',')]  # Convert to 0-based
                i += 2
            elif args[i] == '--text':
                as_text = True
                i += 1
            elif args[i] == '--show':
                show_only = True
                i += 1
            else:
                pdf_path = ' '.join(args[i:])
                break
        
        if not pdf_path:
            print("❌ Please provide a PDF path")
            return
        
        try:
            # Process PDF
            content_items = self.multimodal.encode_pdf(
                pdf_path, 
                pages=pages if not show_only else [0],
                as_images=not as_text
            )
            
            if show_only:
                print(f"📄 Preview of {pdf_path}:")
                if content_items:
                    if content_items[0]['type'] == 'image':
                        display(IPImage(data=content_items[0]['data'], format='png', width=600))
                    else:
                        print(content_items[0]['data'][:1000] + "...")
                return
            
            # Add all content items
            cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
            for item in content_items:
                self.multimodal.add_to_cell(cell_id, item)
            
            if as_text:
                print(f"✅ Extracted text from {len(content_items)} pages")
            else:
                print(f"✅ Added {len(content_items)} page images from PDF")
                # Show thumbnail of first page
                if content_items and content_items[0]['type'] == 'image':
                    print("📄 First page preview:")
                    display(IPImage(data=content_items[0]['data'], format='png', width=200))
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
    
    @line_magic
    def llm_media_clear(self, line):
        """Clear multimodal content from the current cell.
        
        Usage:
            %llm_media_clear              # Clear current cell's media
            %llm_media_clear all          # Clear all cells' media
        """
        if not self.multimodal:
            print("❌ Multimodal support not available.")
            return
        
        if line.strip() == 'all':
            self.multimodal._cell_media.clear()
            print("✅ Cleared all multimodal content")
        else:
            cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
            if cell_id in self.multimodal._cell_media:
                del self.multimodal._cell_media[cell_id]
                print("✅ Cleared multimodal content for current cell")
            else:
                print("ℹ️  No multimodal content in current cell")
    
    @line_magic
    def llm_media_list(self, line):
        """List multimodal content attached to cells.
        
        Usage:
            %llm_media_list               # List all media
            %llm_media_list current       # List current cell's media
        """
        if not self.multimodal:
            print("❌ Multimodal support not available.")
            return
        
        if line.strip() == 'current':
            cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
            media_items = self.multimodal.get_cell_media(cell_id)
            
            if media_items:
                print(f"📎 Media in current cell ({cell_id}):")
                for i, item in enumerate(media_items):
                    print(f"  {i+1}. {item['type']} - {item.get('source', 'embedded')}")
                    if item['type'] == 'image':
                        print(f"     Size: {item.get('size', 'unknown')}")
            else:
                print("ℹ️  No media in current cell")
        else:
            if not self.multimodal._cell_media:
                print("ℹ️  No multimodal content attached to any cells")
                return
            
            print("📎 Multimodal content by cell:")
            for cell_id, items in self.multimodal._cell_media.items():
                print(f"\n{cell_id}: {len(items)} items")
                for i, item in enumerate(items):
                    print(f"  {i+1}. {item['type']} - {item.get('source', 'embedded')}")
    
    @cell_magic
    def llm_vision(self, line, cell):
        """Query a vision-capable LLM with images and text.
        
        Usage:
            %%llm_vision
            What do you see in this image?
            
            %%llm_vision --model=gpt-4o
            Analyze these images and explain...
        """
        if not self.multimodal:
            print("❌ Multimodal support not available.")
            return
        
        # Parse model from line
        model = None
        args = line.strip().split()
        for arg in args:
            if arg.startswith('--model='):
                model = arg.split('=')[1]
        
        if not model:
            model = self.kernel.active_model
        
        # Get media for current cell
        cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
        media_items = self.multimodal.get_cell_media(cell_id)
        
        if not media_items:
            print("❌ No images attached. Use %llm_image or %llm_paste first.")
            return
        
        # Combine media with text query
        content_items = media_items + [{'type': 'text', 'data': cell.strip()}]
        
        # Format for LLM
        messages = self.multimodal.format_for_llm(content_items, model)
        
        # Add previous context if in notebook context mode
        context_messages = self.kernel.get_notebook_cells_as_context()
        if context_messages:
            messages = context_messages + messages
        
        try:
            # Query the LLM
            print(f"🤖 Querying {model} with {len(media_items)} media items...")
            
            # Use the kernel's LLM integration
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Temporarily override messages in the query
            response = loop.run_until_complete(
                self.kernel.llm_integration.query_llm_async(
                    query="",  # Query is in the messages
                    model=model,
                    messages=messages  # Pass our multimodal messages
                )
            )
            
            # Display response
            if self.kernel.display_mode == 'chat':
                html = f'''
                <div style="margin: 10px 0 20px 40px; padding: 10px; background: #f5f5f5; 
                            border-radius: 10px; border-left: 3px solid #2196F3;">
                    <strong>🤖 {model} (vision):</strong><br>
                    <div style="margin-top: 8px; white-space: pre-wrap;">{response}</div>
                </div>
                '''
                display(HTML(html))
            else:
                display(Markdown(response))
            
            # Clear media after use (optional - could make this configurable)
            # self.multimodal._cell_media[cell_id] = []
            
        except Exception as e:
            print(f"❌ Error querying vision model: {e}")
            import traceback
            traceback.print_exc()