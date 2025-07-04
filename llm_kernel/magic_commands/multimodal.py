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
    
    def _get_provider_for_model(self, model: str) -> str:
        """Determine provider from model name."""
        if not model:
            return "openai"
        
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        else:
            return "openai"  # Default
    
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
        pdf_content = None
        text_content = None
        
        if not force_text:
            # Try image first
            img_content = self.multimodal.get_clipboard_image()
            
            # If no image, try PDF
            if not img_content:
                pdf_content = self.multimodal.get_clipboard_pdf()
        
        if not force_image and not img_content and not pdf_content:
            text_content = self.multimodal.get_clipboard_text()
        
        if not img_content and not pdf_content and not text_content:
            print("❌ No content found in clipboard")
            return
        
        if show_only:
            if img_content:
                print("📋 Clipboard contains an image:")
                # Convert base64 string to bytes for IPython display
                import base64
                img_bytes = base64.b64decode(img_content['data'])
                display(IPImage(data=img_bytes, format='png'))
            elif pdf_content:
                print(f"📋 Clipboard contains a PDF file path: {pdf_content['filename']}")
                print(f"   Size: {pdf_content['size'] / (1024*1024):.2f} MB")
                print(f"   Path: {pdf_content['path']}")
            elif text_content:
                print("📋 Clipboard contains text:")
                print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
            return
        
        # Add to current cell's multimodal content
        cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
        
        if img_content:
            # Add image to conversation history as a multimodal message
            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_content['data']}"
                        }
                    }
                ]
            }
            self.kernel.conversation_history.append(image_message)
            
            # Also add to cell for immediate use
            self.multimodal.add_to_cell(cell_id, img_content)
            
            print(f"✅ Pasted image ({img_content['size'][0]}x{img_content['size'][1]}) - added to conversation context")
            # Show thumbnail - convert base64 to bytes
            import base64
            img_bytes = base64.b64decode(img_content['data'])
            display(IPImage(data=img_bytes, format='png', width=200))
        elif pdf_content:
            # Upload PDF using the file upload manager
            print(f"📄 Uploading PDF '{pdf_content['filename']}'...")
            
            try:
                # Get or create file upload manager
                if not hasattr(self.kernel, 'file_upload_manager'):
                    from ..file_upload_manager import FileUploadManager
                    self.kernel.file_upload_manager = FileUploadManager(self.kernel.log)
                    
                    # Initialize with LLM clients if available
                    if hasattr(self.kernel, 'llm_integration'):
                        for provider, client in self.kernel.llm_integration._get_provider_clients().items():
                            self.kernel.file_upload_manager.set_provider_client(provider, client)
                
                # Determine provider based on active model
                provider = self._get_provider_for_model(self.kernel.active_model)
                
                # Upload the file
                upload_info = self.kernel.file_upload_manager.upload_file(
                    pdf_content['path'],
                    purpose="assistants",
                    provider=provider
                )
                
                if upload_info:
                    # Add file reference to conversation
                    file_ref = self.kernel.file_upload_manager.format_file_reference(upload_info, provider)
                    
                    # Add metadata for persistence
                    file_ref['_llm_kernel_meta'] = {
                        'file_hash': upload_info['file_hash'],
                        'provider': provider,
                        'cached_path': upload_info.get('cached_path'),
                        'original_name': pdf_content['filename']
                    }
                    
                    pdf_message = {
                        "role": "user",
                        "content": [
                            file_ref,
                            {
                                "type": "text",
                                "text": f"[Uploaded PDF: {pdf_content['filename']}]"
                            }
                        ]
                    }
                    self.kernel.conversation_history.append(pdf_message)
                    
                    print(f"✅ Uploaded PDF '{pdf_content['filename']}' ({pdf_content['size'] / (1024*1024):.2f} MB)")
                    print(f"📎 File ID: {upload_info.get('file_id', 'embedded')}")
                    print("💡 You can now ask questions about this PDF in any cell")
                    
                    # Track uploaded file
                    if not hasattr(self.kernel, '_uploaded_files'):
                        self.kernel._uploaded_files = []
                    
                    self.kernel._uploaded_files.append({
                        'filename': pdf_content['filename'],
                        'path': pdf_content['path'],
                        'size': pdf_content['size'],
                        'type': 'pdf',
                        'source': 'clipboard',
                        'upload_info': upload_info
                    })
                
            except Exception as e:
                print(f"❌ Error uploading PDF: {e}")
                print("💡 Falling back to image conversion...")
                
                # Fallback to image conversion
                try:
                    pdf_pages = self.multimodal.encode_pdf(pdf_content['path'], as_images=True)
                    for page_img in pdf_pages:
                        if page_img['type'] == 'image':
                            image_message = {
                                "role": "user",
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{page_img['data']}"}
                                }]
                            }
                            self.kernel.conversation_history.append(image_message)
                    print(f"✅ Converted PDF to {len(pdf_pages)} images as fallback")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
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
                import base64
                img_bytes = base64.b64decode(img_content['data'])
                display(IPImage(data=img_bytes, format='png', width=400))
                return
            
            # Add image to conversation history as a multimodal message
            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_content['data']}"
                        }
                    }
                ]
            }
            self.kernel.conversation_history.append(image_message)
            
            # Also add to current cell's multimodal content
            cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
            self.multimodal.add_to_cell(cell_id, img_content)
            
            print(f"✅ Added image ({img_content['size'][0]}x{img_content['size'][1]}) - added to conversation context")
            # Show thumbnail
            import base64
            img_bytes = base64.b64decode(img_content['data'])
            display(IPImage(data=img_bytes, format='png', width=200))
            
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
                        import base64
                        img_bytes = base64.b64decode(content_items[0]['data'])
                        display(IPImage(data=img_bytes, format='png', width=600))
                    else:
                        print(content_items[0]['data'][:1000] + "...")
                return
            
            # Add all content items
            cell_id = getattr(self.kernel, '_current_cell_id', 'unknown')
            for item in content_items:
                self.multimodal.add_to_cell(cell_id, item)
                
                # Add to conversation history
                if item['type'] == 'image':
                    # Add images as multimodal messages
                    image_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{item['data']}"
                                }
                            }
                        ]
                    }
                    self.kernel.conversation_history.append(image_message)
                elif item['type'] == 'text':
                    # Add extracted text as regular message
                    text_message = {
                        "role": "user",
                        "content": f"[PDF Text from {pdf_path}]:\n{item['data']}"
                    }
                    self.kernel.conversation_history.append(text_message)
            
            if as_text:
                print(f"✅ Extracted text from {len(content_items)} pages")
            else:
                print(f"✅ Added {len(content_items)} page images from PDF")
                # Show thumbnail of first page
                if content_items and content_items[0]['type'] == 'image':
                    print("📄 First page preview:")
                    import base64
                    img_bytes = base64.b64decode(content_items[0]['data'])
                    display(IPImage(data=img_bytes, format='png', width=200))
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
    
    @line_magic
    def llm_media_clear(self, line):
        """Clear multimodal content from the current cell or conversation.
        
        Usage:
            %llm_media_clear              # Clear current cell's media
            %llm_media_clear all          # Clear all cells' media
            %llm_media_clear history      # Clear images from conversation history
        """
        if not self.multimodal:
            print("❌ Multimodal support not available.")
            return
        
        if line.strip() == 'all':
            self.multimodal._cell_media.clear()
            print("✅ Cleared all multimodal content from cells")
        elif line.strip() == 'history':
            # Remove multimodal messages from conversation history
            if hasattr(self.kernel, 'conversation_history'):
                original_len = len(self.kernel.conversation_history)
                self.kernel.conversation_history = [
                    msg for msg in self.kernel.conversation_history 
                    if not (isinstance(msg.get('content'), list) and 
                           any(item.get('type') == 'image_url' for item in msg['content']))
                ]
                removed = original_len - len(self.kernel.conversation_history)
                print(f"✅ Removed {removed} image messages from conversation history")
            else:
                print("ℹ️  No conversation history found")
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
    
    @line_magic
    def llm_cache_info(self, line):
        """Show information about the file cache.
        
        Usage:
            %llm_cache_info    # Show cache statistics
        """
        if hasattr(self.kernel, 'file_upload_manager'):
            info = self.kernel.file_upload_manager.get_cache_info()
            print("📁 LLM Kernel File Cache")
            print(f"   Location: {info['cache_dir']}")
            print(f"   Files: {info['num_files']}")
            print(f"   Size: {info['total_size_mb']:.2f} MB")
            print(f"   Types: {info['file_types']}")
        else:
            print("ℹ️  No file cache initialized")
    
    @line_magic
    def llm_cache_list(self, line):
        """List cached files.
        
        Usage:
            %llm_cache_list         # List all cached files
            %llm_cache_list pdf     # List only PDFs
            %llm_cache_list image   # List only images
        """
        if not hasattr(self.kernel, 'file_upload_manager'):
            print("ℹ️  No file cache initialized")
            return
        
        file_type = line.strip() if line.strip() else None
        files = self.kernel.file_upload_manager.list_cached_files()
        
        if file_type:
            files = [f for f in files if f.get('file_type') == file_type]
        
        if not files:
            print(f"ℹ️  No {'files' if not file_type else file_type + ' files'} in cache")
            return
        
        print(f"📁 Cached {'files' if not file_type else file_type + ' files'}:")
        for i, file_info in enumerate(files, 1):
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"\n{i}. {file_info['original_name']}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Cached: {file_info['cached_at']}")
            print(f"   Hash: {file_info['file_hash'][:16]}...")
            
            # Show upload history
            if file_info.get('upload_history'):
                print("   Uploads:")
                for upload in file_info['upload_history']:
                    provider = upload['provider']
                    file_id = upload['upload_info'].get('file_id', 'embedded')
                    print(f"     - {provider}: {file_id}")
    
    @line_magic
    def llm_cache_clear(self, line):
        """Clear the file cache.
        
        Usage:
            %llm_cache_clear            # Clear all cached files
            %llm_cache_clear --days=7   # Clear files older than 7 days
        """
        if not hasattr(self.kernel, 'file_upload_manager'):
            print("ℹ️  No file cache initialized")
            return
        
        older_than_days = None
        if '--days=' in line:
            try:
                days_str = line.split('--days=')[1].split()[0]
                older_than_days = int(days_str)
            except:
                print("❌ Invalid days parameter")
                return
        
        # Confirm before clearing
        if older_than_days:
            confirm = input(f"Clear cached files older than {older_than_days} days? (y/N): ")
        else:
            confirm = input("Clear ALL cached files? (y/N): ")
        
        if confirm.lower() == 'y':
            count = self.kernel.file_upload_manager.cache_manager.clear_cache(older_than_days)
            print(f"✅ Cleared {count} cached files")
        else:
            print("❌ Cache clear cancelled")