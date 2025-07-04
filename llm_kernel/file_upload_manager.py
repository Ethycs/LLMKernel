"""
File Upload Manager for LLM APIs

Handles file uploads to various LLM providers (OpenAI, Anthropic, etc.)
and manages file references for conversations.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True  
except ImportError:
    HAS_ANTHROPIC = False


from .file_cache_manager import FileCacheManager


class FileUploadManager:
    """Manages file uploads to LLM providers."""
    
    def __init__(self, logger=None, cache_dir: str = ".llm_kernel_cache"):
        self.log = logger
        self._uploaded_files = {}  # file_hash -> upload_info
        self._provider_clients = {}
        self.cache_manager = FileCacheManager(cache_dir, logger)
        
    def set_provider_client(self, provider: str, client: Any):
        """Set the client for a specific provider."""
        self._provider_clients[provider] = client
        
    def get_file_hash(self, file_path: str) -> str:
        """Get hash of file to check if already uploaded."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def upload_file(self, file_path: str, purpose: str = "assistants", provider: str = "openai") -> Optional[Dict[str, Any]]:
        """
        Upload a file to the specified provider.
        
        Args:
            file_path: Path to the file
            purpose: Purpose of upload (e.g., "assistants" for OpenAI)
            provider: Which provider to upload to
            
        Returns:
            Upload info including file_id
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Cache the file locally first
        cache_info = self.cache_manager.cache_file(str(file_path))
        if not cache_info:
            raise RuntimeError(f"Failed to cache file: {file_path}")
            
        file_hash = cache_info['file_hash']
        
        # Check if we have a recent upload to this provider
        existing_upload = self.cache_manager.find_upload(provider, file_hash)
        if existing_upload:
            if self.log:
                self.log.debug(f"Using existing upload for {file_path.name}")
            return existing_upload
        
        # Use cached file for upload
        cached_path = Path(cache_info['cached_path'])
        
        # Upload based on provider
        if provider == "openai":
            upload_info = self._upload_to_openai(cached_path, purpose)
        elif provider == "anthropic":
            upload_info = self._upload_to_anthropic(cached_path)
        elif provider == "gemini" or provider == "google":
            upload_info = self._upload_to_gemini(cached_path)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        if upload_info:
            # Record the upload (convert datetime to string for JSON serialization)
            upload_info['uploaded_at'] = datetime.now().isoformat()
            upload_info['file_hash'] = file_hash
            upload_info['original_path'] = str(file_path)
            upload_info['cached_path'] = str(cached_path)
            
            # Save to cache manager
            self.cache_manager.record_upload(file_hash, provider, upload_info)
            
        return upload_info
    
    def _upload_to_openai(self, file_path: Path, purpose: str) -> Optional[Dict[str, Any]]:
        """Upload file to OpenAI."""
        if not HAS_OPENAI:
            if self.log:
                self.log.error("OpenAI library not installed")
            return None
            
        client = self._provider_clients.get('openai')
        if not client:
            if self.log:
                self.log.error("OpenAI client not configured")
            return None
            
        try:
            # Upload file to OpenAI
            with open(file_path, 'rb') as f:
                # Use "user_data" purpose for model inputs as recommended
                file_obj = client.files.create(
                    file=f,
                    purpose="user_data"  # Changed from "assistants" to "user_data"
                )
            
            if self.log:
                self.log.info(f"Uploaded file to OpenAI: {file_obj.id}")
                
            return {
                'file_id': file_obj.id,
                'filename': file_path.name,
                'size': file_obj.bytes,
                'provider': 'openai',
                'purpose': purpose,
                'created_at': file_obj.created_at if isinstance(file_obj.created_at, str) else datetime.fromtimestamp(file_obj.created_at).isoformat()
            }
            
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to upload file to OpenAI: {e}")
            
            # Fall back to storing file data for conversion
            with open(file_path, 'rb') as f:
                file_data = f.read()
                
            return {
                'file_data': file_data,
                'filename': file_path.name,
                'size': len(file_data),
                'provider': 'openai',
                'purpose': purpose,
                'needs_conversion': True
            }
    
    def _upload_to_anthropic(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Upload file to Anthropic."""
        # Try using the new Files API first (for Claude 3.5+ models)
        if HAS_ANTHROPIC:
            try:
                from .anthropic_file_integration import AnthropicFileIntegration
                anthropic_files = AnthropicFileIntegration(logger=self.log)
                
                result = anthropic_files.upload_file(str(file_path))
                if result and 'file_id' in result:
                    if self.log:
                        self.log.info(f"Successfully uploaded to Anthropic Files API: {result['file_id']}")
                    return result
            except Exception as e:
                if self.log:
                    self.log.debug(f"Anthropic Files API not available or failed: {e}")
        
        # Fallback to embedding file data directly (for older models or if Files API fails)
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        return {
            'file_data': file_data,
            'filename': file_path.name,
            'size': len(file_data),
            'provider': 'anthropic',
            'mime_type': self._get_mime_type(file_path),
            'embed_directly': True  # Flag to indicate this should be embedded in messages
        }
    
    def _upload_to_gemini(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Upload file to Google Gemini."""
        try:
            from .gemini_file_integration import GeminiFileIntegration
            gemini_files = GeminiFileIntegration(logger=self.log)
            
            result = gemini_files.upload_file(str(file_path))
            if result:
                if self.log:
                    self.log.info(f"Successfully uploaded to Gemini: {result['file_name']}")
                return result
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to upload to Gemini: {e}")
        
        # No fallback for Gemini - it should support files directly
        return None
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'
    
    def _is_upload_valid(self, upload_info: Dict[str, Any]) -> bool:
        """Check if an upload is still valid."""
        # OpenAI files expire after a certain time
        uploaded_at = upload_info.get('uploaded_at')
        if not uploaded_at:
            return False
            
        # Consider uploads valid for 1 hour (adjust as needed)
        age = datetime.now() - uploaded_at
        return age < timedelta(hours=1)
    
    def format_file_reference(self, upload_info: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Format file reference for inclusion in messages.
        
        Args:
            upload_info: The upload info from upload_file()
            provider: The provider to format for
            
        Returns:
            Formatted file reference for the provider's API
        """
        if provider == "openai":
            # OpenAI expects file_id in the message
            return {
                "type": "file",
                "file": {
                    "file_id": upload_info['file_id']
                }
            }
        elif provider == "anthropic":
            # Check if we have a file_id (new Files API) or need to embed directly
            if 'file_id' in upload_info:
                # Use the Files API format
                return {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": upload_info['file_id']
                    }
                }
            else:
                # Fallback to embedding file data directly
                import base64
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": upload_info['mime_type'],
                        "data": base64.b64encode(upload_info['file_data']).decode('utf-8')
                    }
                }
        elif provider == "gemini" or provider == "google":
            # Gemini uses the file object directly, not in message format
            # This is handled differently in the LLM integration
            return {
                "type": "file",
                "file_name": upload_info['file_name'],
                "provider": "gemini"
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def delete_file(self, file_id: str, provider: str = "openai"):
        """Delete an uploaded file."""
        if provider == "openai":
            client = self._provider_clients.get('openai')
            if client:
                try:
                    client.files.delete(file_id)
                    # Remove from cache
                    self._uploaded_files = {
                        k: v for k, v in self._uploaded_files.items()
                        if v.get('file_id') != file_id
                    }
                except Exception as e:
                    if self.log:
                        self.log.error(f"Failed to delete file: {e}")
    
    def list_uploaded_files(self) -> List[Dict[str, Any]]:
        """List all uploaded files."""
        return list(self._uploaded_files.values())
    
    def clear_cache(self):
        """Clear the upload cache (doesn't delete files from providers)."""
        self._uploaded_files.clear()
    
    def restore_from_cache(self, file_hash: str, provider: str) -> Optional[Dict[str, Any]]:
        """
        Restore a file from cache and re-upload if needed.
        
        Used when loading a notebook that references files.
        """
        # Get cached file
        cached_path = self.cache_manager.get_cached_file(file_hash)
        if not cached_path:
            return None
            
        # Check for existing upload
        existing_upload = self.cache_manager.find_upload(provider, file_hash)
        if existing_upload:
            return existing_upload
            
        # Re-upload from cache
        if self.log:
            self.log.info(f"Re-uploading cached file: {cached_path.name}")
            
        return self.upload_file(str(cached_path), provider=provider)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the file cache."""
        return self.cache_manager.get_cache_info()
    
    def list_cached_files(self) -> List[Dict[str, Any]]:
        """List all cached files."""
        return self.cache_manager.list_cached_files()