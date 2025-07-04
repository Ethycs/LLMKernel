"""
File Cache Manager for LLM Kernel

Manages local caching of uploaded files (PDFs, images) for persistence
across notebook sessions.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List


class FileCacheManager:
    """Manages local file cache for multimodal content."""
    
    def __init__(self, cache_dir: str = ".llm_kernel_cache", logger=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger
        
        # Create subdirectories
        self.files_dir = self.cache_dir / "files"
        self.files_dir.mkdir(exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load cache metadata: {e}")
        return {"files": {}, "uploads": {}}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save cache metadata: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Get SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def cache_file(self, file_path: str, file_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Cache a file locally and return cache info.
        
        Args:
            file_path: Path to the file to cache
            file_type: Type of file (pdf, image, etc.)
            
        Returns:
            Cache info including cached path and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        # Get file hash
        file_hash = self.get_file_hash(str(file_path))
        
        # Check if already cached
        if file_hash in self.metadata.get("files", {}):
            cache_info = self.metadata["files"][file_hash]
            cached_path = Path(cache_info["cached_path"])
            
            # Verify cached file still exists
            if cached_path.exists():
                if self.logger:
                    self.logger.debug(f"File already cached: {file_path.name}")
                return cache_info
        
        # Determine file type
        if file_type is None:
            ext = file_path.suffix.lower()
            if ext == '.pdf':
                file_type = 'pdf'
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                file_type = 'image'
            else:
                file_type = 'other'
        
        # Create cached filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cached_filename = f"{timestamp}_{file_hash[:8]}_{file_path.name}"
        cached_path = self.files_dir / cached_filename
        
        # Copy file to cache
        try:
            shutil.copy2(file_path, cached_path)
            
            # Create cache info
            cache_info = {
                "original_path": str(file_path),
                "original_name": file_path.name,
                "cached_path": str(cached_path),
                "file_hash": file_hash,
                "file_type": file_type,
                "size": file_path.stat().st_size,
                "cached_at": datetime.now().isoformat(),
                "upload_history": []  # Track upload IDs for different providers
            }
            
            # Save to metadata
            self.metadata["files"][file_hash] = cache_info
            self._save_metadata()
            
            if self.logger:
                self.logger.info(f"Cached file: {file_path.name} -> {cached_filename}")
            
            return cache_info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to cache file: {e}")
            return None
    
    def get_cached_file(self, file_hash: str) -> Optional[Path]:
        """Get path to cached file by hash."""
        if file_hash in self.metadata.get("files", {}):
            cached_path = Path(self.metadata["files"][file_hash]["cached_path"])
            if cached_path.exists():
                return cached_path
        return None
    
    def record_upload(self, file_hash: str, provider: str, upload_info: Dict[str, Any]):
        """Record that a file was uploaded to a provider."""
        if file_hash in self.metadata.get("files", {}):
            upload_record = {
                "provider": provider,
                "upload_info": upload_info,
                "uploaded_at": datetime.now().isoformat()
            }
            
            # Add to upload history
            self.metadata["files"][file_hash]["upload_history"].append(upload_record)
            
            # Also track by provider and upload ID for quick lookup
            if "uploads" not in self.metadata:
                self.metadata["uploads"] = {}
            
            upload_id = upload_info.get("file_id") or upload_info.get("id")
            if upload_id:
                self.metadata["uploads"][f"{provider}:{upload_id}"] = {
                    "file_hash": file_hash,
                    "upload_info": upload_info,
                    "uploaded_at": upload_record["uploaded_at"]
                }
            
            self._save_metadata()
    
    def find_upload(self, provider: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find a recent upload for a file to a specific provider."""
        if file_hash in self.metadata.get("files", {}):
            upload_history = self.metadata["files"][file_hash].get("upload_history", [])
            
            # Find most recent upload to this provider
            for upload in reversed(upload_history):
                if upload["provider"] == provider:
                    # Check if upload is recent (within last hour)
                    uploaded_at = datetime.fromisoformat(upload["uploaded_at"])
                    age = datetime.now() - uploaded_at
                    if age.total_seconds() < 3600:  # 1 hour
                        return upload["upload_info"]
        
        return None
    
    def list_cached_files(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all cached files, optionally filtered by type."""
        files = []
        for file_hash, cache_info in self.metadata.get("files", {}).items():
            if file_type is None or cache_info.get("file_type") == file_type:
                # Check if cached file still exists
                if Path(cache_info["cached_path"]).exists():
                    files.append(cache_info)
        
        # Sort by cached_at timestamp
        files.sort(key=lambda x: x.get("cached_at", ""), reverse=True)
        return files
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear cached files, optionally only those older than specified days."""
        cleared_count = 0
        
        for file_hash, cache_info in list(self.metadata.get("files", {}).items()):
            cached_path = Path(cache_info["cached_path"])
            
            # Check age if specified
            if older_than_days is not None:
                cached_at = datetime.fromisoformat(cache_info["cached_at"])
                age = datetime.now() - cached_at
                if age.days < older_than_days:
                    continue
            
            # Remove file
            if cached_path.exists():
                try:
                    cached_path.unlink()
                    cleared_count += 1
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to delete cached file: {e}")
            
            # Remove from metadata
            del self.metadata["files"][file_hash]
        
        # Clean up uploads metadata
        self.metadata["uploads"] = {
            k: v for k, v in self.metadata.get("uploads", {}).items()
            if v["file_hash"] in self.metadata["files"]
        }
        
        self._save_metadata()
        
        if self.logger:
            self.logger.info(f"Cleared {cleared_count} cached files")
        
        return cleared_count
    
    def get_cache_size(self) -> int:
        """Get total size of cached files in bytes."""
        total_size = 0
        for cache_info in self.metadata.get("files", {}).values():
            cached_path = Path(cache_info["cached_path"])
            if cached_path.exists():
                total_size += cached_path.stat().st_size
        return total_size
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(self.metadata.get("files", {})),
            "total_size": self.get_cache_size(),
            "total_size_mb": self.get_cache_size() / (1024 * 1024),
            "file_types": self._count_file_types()
        }
    
    def _count_file_types(self) -> Dict[str, int]:
        """Count cached files by type."""
        counts = {}
        for cache_info in self.metadata.get("files", {}).values():
            file_type = cache_info.get("file_type", "other")
            counts[file_type] = counts.get(file_type, 0) + 1
        return counts