"""
Anthropic Files API Integration

This module provides integration with Anthropic's Files API for native
file uploads including PDFs (Claude 3.5+ models).
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import anthropic
    import httpx
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None
    httpx = None


class AnthropicFileIntegration:
    """Handles Anthropic Files API for file uploads."""
    
    def __init__(self, logger=None):
        self.log = logger
        self._client = None
        self._file_cache = {}  # file_hash -> file_id mapping
        
        if HAS_ANTHROPIC and os.getenv('ANTHROPIC_API_KEY'):
            self._client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                default_headers={
                    "anthropic-beta": "files-api-2025-04-14"
                }
            )
    
    def upload_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Upload a file to Anthropic Files API."""
        if not self._client:
            if self.log:
                self.log.warning("Anthropic client not configured")
            return None
        
        file_path = Path(file_path)
        if not file_path.exists():
            if self.log:
                self.log.error(f"File not found: {file_path}")
            return None
        
        try:
            # Use httpx for file upload as the anthropic SDK might not have full support yet
            import httpx
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, self._get_mime_type(file_path))}
                
                response = httpx.post(
                    "https://api.anthropic.com/v1/files",
                    headers={
                        "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "files-api-2025-04-14"
                    },
                    files=files
                )
            
            if response.status_code == 200:
                file_data = response.json()
                file_id = file_data.get('id')
                
                if self.log:
                    self.log.info(f"Uploaded file to Anthropic: {file_id}")
                
                return {
                    'file_id': file_id,
                    'filename': file_path.name,
                    'size': file_path.stat().st_size,
                    'provider': 'anthropic',
                    'mime_type': file_data.get('mime_type'),
                    'created_at': file_data.get('created_at')
                }
            else:
                if self.log:
                    self.log.error(f"Failed to upload to Anthropic: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            if self.log:
                self.log.error(f"Error uploading file to Anthropic: {e}")
            return None
    
    def list_files(self) -> Optional[List[Dict[str, Any]]]:
        """List all uploaded files."""
        if not self._client:
            return None
        
        try:
            response = httpx.get(
                "https://api.anthropic.com/v1/files",
                headers={
                    "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "files-api-2025-04-14"
                }
            )
            
            if response.status_code == 200:
                return response.json().get('data', [])
            else:
                if self.log:
                    self.log.error(f"Failed to list files: {response.status_code}")
                return None
                
        except Exception as e:
            if self.log:
                self.log.error(f"Error listing files: {e}")
            return None
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Anthropic."""
        if not self._client:
            return False
        
        try:
            response = httpx.delete(
                f"https://api.anthropic.com/v1/files/{file_id}",
                headers={
                    "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "files-api-2025-04-14"
                }
            )
            
            if response.status_code in [200, 204]:
                if self.log:
                    self.log.info(f"Deleted file from Anthropic: {file_id}")
                return True
            else:
                if self.log:
                    self.log.error(f"Failed to delete file: {response.status_code}")
                return False
                
        except Exception as e:
            if self.log:
                self.log.error(f"Error deleting file: {e}")
            return False
    
    def format_file_for_message(self, file_id: str, file_type: str = "document") -> Dict[str, Any]:
        """Format file reference for use in Claude messages."""
        if file_type == "document":
            return {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id
                }
            }
        elif file_type == "image":
            return {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": file_id
                }
            }
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'