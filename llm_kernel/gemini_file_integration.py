"""
Google Gemini Files API Integration

This module provides integration with Google's Gemini Files API for native
file uploads including PDFs.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None


class GeminiFileIntegration:
    """Handles Google Gemini Files API for file uploads."""
    
    def __init__(self, logger=None):
        self.log = logger
        self._configured = False
        
        if HAS_GENAI and os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self._configured = True
    
    def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Upload a file to Gemini Files API."""
        if not self._configured:
            if self.log:
                self.log.warning("Gemini not configured - need GOOGLE_API_KEY")
            return None
        
        file_path = Path(file_path)
        if not file_path.exists():
            if self.log:
                self.log.error(f"File not found: {file_path}")
            return None
        
        try:
            # Upload the file
            uploaded_file = genai.upload_file(
                path=str(file_path),
                display_name=display_name or file_path.name
            )
            
            # Wait for processing to complete
            while uploaded_file.state.name == 'PROCESSING':
                time.sleep(0.5)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            if uploaded_file.state.name == 'FAILED':
                if self.log:
                    self.log.error(f"File processing failed: {uploaded_file.error}")
                return None
            
            if self.log:
                self.log.info(f"Uploaded file to Gemini: {uploaded_file.name}")
            
            return {
                'file_name': uploaded_file.name,  # Gemini uses 'name' not 'file_id'
                'display_name': uploaded_file.display_name,
                'filename': file_path.name,
                'size': uploaded_file.size_bytes,
                'provider': 'gemini',
                'mime_type': uploaded_file.mime_type,
                'uri': uploaded_file.uri,
                'created_at': uploaded_file.create_time.isoformat() if hasattr(uploaded_file.create_time, 'isoformat') else str(uploaded_file.create_time)
            }
            
        except Exception as e:
            if self.log:
                self.log.error(f"Error uploading file to Gemini: {e}")
            return None
    
    def list_files(self) -> Optional[List[Dict[str, Any]]]:
        """List all uploaded files."""
        if not self._configured:
            return None
        
        try:
            files = []
            for file in genai.list_files():
                files.append({
                    'name': file.name,
                    'display_name': file.display_name,
                    'mime_type': file.mime_type,
                    'size': file.size_bytes,
                    'created_at': str(file.create_time),
                    'state': file.state.name
                })
            return files
            
        except Exception as e:
            if self.log:
                self.log.error(f"Error listing files: {e}")
            return None
    
    def delete_file(self, file_name: str) -> bool:
        """Delete a file from Gemini."""
        if not self._configured:
            return False
        
        try:
            genai.delete_file(file_name)
            if self.log:
                self.log.info(f"Deleted file from Gemini: {file_name}")
            return True
            
        except Exception as e:
            if self.log:
                self.log.error(f"Error deleting file: {e}")
            return False
    
    def get_file(self, file_name: str) -> Optional[Any]:
        """Get file object for use in generation."""
        if not self._configured:
            return None
        
        try:
            return genai.get_file(file_name)
        except Exception as e:
            if self.log:
                self.log.error(f"Error getting file: {e}")
            return None