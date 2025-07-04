"""
Native PDF support for LLM APIs that accept direct PDF uploads.
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import display, Markdown


@magics_class
class NativePDFMagics(Magics):
    """Magic commands for native PDF upload support."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
    
    @line_magic
    def llm_pdf_native(self, line):
        """Upload PDF directly to LLM conversation (for APIs that support native PDF).
        
        This uses the modern approach where PDFs are uploaded directly without
        converting to images first. Supported by OpenAI GPT-4.1+, Claude, etc.
        
        Usage:
            %llm_pdf_native document.pdf              # Upload entire PDF
            %llm_pdf_native --preview document.pdf    # Show preview of what will be uploaded
            
        Note: This is different from %llm_pdf which converts pages to images.
        """
        args = line.strip().split()
        if not args:
            print("❌ Please provide a PDF path")
            return
        
        preview_only = '--preview' in args
        if preview_only:
            args.remove('--preview')
        
        pdf_path = ' '.join(args)
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return
        
        if not pdf_path.suffix.lower() == '.pdf':
            print(f"❌ Not a PDF file: {pdf_path}")
            return
        
        try:
            # Read PDF file
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            file_size_mb = len(pdf_bytes) / (1024 * 1024)
            
            if preview_only:
                print(f"📄 PDF: {pdf_path.name}")
                print(f"   Size: {file_size_mb:.2f} MB")
                print(f"   Full path: {pdf_path.absolute()}")
                return
            
            # Determine the provider based on the active model
            provider = self._get_provider_from_model(self.kernel.active_model if hasattr(self.kernel, 'active_model') else '')
            
            # Check if we should use file upload API
            if provider in ['openai', 'anthropic', 'gemini']:
                # Use file upload manager for OpenAI
                if not hasattr(self.kernel, 'file_upload_manager'):
                    from ..file_upload_manager import FileUploadManager
                    self.kernel.file_upload_manager = FileUploadManager(logger=self.kernel.log)
                    # Set the OpenAI client
                    providers = self.kernel.llm_integration._get_provider_clients()
                    for provider, client in providers.items():
                        self.kernel.file_upload_manager.set_provider_client(provider, client)
                
                # Upload the file
                upload_info = self.kernel.file_upload_manager.upload_file(
                    str(pdf_path),
                    purpose="assistants",
                    provider=provider
                )
                
                if upload_info and ('file_id' in upload_info or 'file_name' in upload_info):
                    # Create a message with the file reference
                    if provider == 'openai':
                        # Use the correct format for OpenAI
                        file_ref = {
                            "type": "file",
                            "file": {
                                "file_id": upload_info['file_id']
                            }
                        }
                        print(f"✅ Uploaded PDF '{pdf_path.name}' to OpenAI (file_id: {upload_info['file_id']})")
                    elif provider == 'anthropic':
                        file_ref = {
                            "type": "document",
                            "source": {
                                "type": "file",
                                "file_id": upload_info['file_id']
                            }
                        }
                        print(f"✅ Uploaded PDF '{pdf_path.name}' to Claude (file_id: {upload_info['file_id']})")
                    elif provider == 'gemini':
                        file_ref = {
                            "type": "file",
                            "file": {
                                "file_name": upload_info['file_name'],
                                "filename": pdf_path.name,
                                "provider": "gemini"
                            }
                        }
                        print(f"✅ Uploaded PDF '{pdf_path.name}' to Gemini (file: {upload_info['file_name']})")
                    
                    pdf_message = {
                        "role": "user",
                        "content": [
                            file_ref,
                            {
                                "type": "text", 
                                "text": f"[Uploaded PDF: {pdf_path.name}]"
                            }
                        ]
                    }
                else:
                    # Fallback to base64 encoding
                    pdf_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "filename": pdf_path.name,
                                    "file_data": f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('utf-8')}",
                                    "file_size": len(pdf_bytes)
                                }
                            },
                            {
                                "type": "text", 
                                "text": f"[Uploaded PDF: {pdf_path.name}]"
                            }
                        ]
                    }
                    print(f"✅ Embedded PDF '{pdf_path.name}' ({file_size_mb:.2f} MB) in conversation")
            else:
                # For non-OpenAI models, use base64 encoding
                pdf_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": pdf_path.name,
                                "file_data": f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('utf-8')}",
                                "file_size": len(pdf_bytes)
                            }
                        },
                        {
                            "type": "text", 
                            "text": f"[Uploaded PDF: {pdf_path.name}]"
                        }
                    ]
                }
            
            # Add to conversation history
            self.kernel.conversation_history.append(pdf_message)
            
            if 'gpt' not in self.kernel.active_model.lower():
                print(f"✅ Embedded PDF '{pdf_path.name}' ({file_size_mb:.2f} MB) in conversation")
            
            print("💡 You can now ask questions about this PDF in any cell")
            
            # Also store metadata for tracking
            if not hasattr(self.kernel, '_uploaded_files'):
                self.kernel._uploaded_files = []
            
            self.kernel._uploaded_files.append({
                'filename': pdf_path.name,
                'path': str(pdf_path.absolute()),
                'size': len(pdf_bytes),
                'type': 'pdf'
            })
            
        except Exception as e:
            print(f"❌ Error uploading PDF: {e}")
    
    @line_magic
    def llm_files_list(self, line):
        """List files uploaded to the conversation.
        
        Usage:
            %llm_files_list    # Show all uploaded files
        """
        if not hasattr(self.kernel, '_uploaded_files') or not self.kernel._uploaded_files:
            print("ℹ️  No files uploaded to conversation")
            return
        
        print("📁 Uploaded files in conversation:")
        for i, file_info in enumerate(self.kernel._uploaded_files, 1):
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"  {i}. {file_info['filename']} ({file_info['type']}, {size_mb:.2f} MB)")
            if 'path' in file_info:
                print(f"      Path: {file_info['path']}")
    
    @line_magic
    def llm_files_clear(self, line):
        """Clear uploaded files from conversation history.
        
        Usage:
            %llm_files_clear    # Remove all file uploads from conversation
        """
        if hasattr(self.kernel, 'conversation_history'):
            original_len = len(self.kernel.conversation_history)
            
            # Remove messages with file content
            self.kernel.conversation_history = [
                msg for msg in self.kernel.conversation_history 
                if not (isinstance(msg.get('content'), list) and 
                       any(item.get('type') == 'file' for item in msg['content']))
            ]
            
            removed = original_len - len(self.kernel.conversation_history)
            
            # Clear file tracking
            if hasattr(self.kernel, '_uploaded_files'):
                self.kernel._uploaded_files = []
            
            print(f"✅ Removed {removed} file uploads from conversation history")
        else:
            print("ℹ️  No conversation history found")
    
    def _get_provider_from_model(self, model_name: str) -> str:
        """Determine the provider from the model name."""
        if not model_name:
            return "unknown"
        
        model_lower = model_name.lower()
        if 'gpt' in model_lower or 'o3' in model_lower:
            return 'openai'
        elif 'claude' in model_lower:
            return 'anthropic'
        elif 'gemini' in model_lower or 'bison' in model_lower:
            return 'gemini'
        else:
            return 'unknown'