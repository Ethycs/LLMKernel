"""
Platform-specific clipboard utilities for handling file copies.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import base64


def get_clipboard_files() -> Optional[List[Dict[str, Any]]]:
    """
    Get files from clipboard that were copied with Ctrl+C.
    
    Returns:
        List of file info dicts or None if no files in clipboard
    """
    if sys.platform == 'win32':
        return _get_windows_clipboard_files()
    elif sys.platform == 'darwin':
        return _get_mac_clipboard_files()
    else:
        return _get_linux_clipboard_files()


def _get_windows_clipboard_files() -> Optional[List[Dict[str, Any]]]:
    """Get files from Windows clipboard."""
    try:
        import win32clipboard
        import win32con
        
        win32clipboard.OpenClipboard()
        try:
            # Check if clipboard contains file drop list
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_HDROP):
                file_paths = win32clipboard.GetClipboardData(win32con.CF_HDROP)
                
                files = []
                for file_path in file_paths:
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        file_info = {
                            'path': str(path),
                            'filename': path.name,
                            'size': path.stat().st_size,
                            'extension': path.suffix.lower()
                        }
                        
                        # Read file content for supported types
                        if file_info['extension'] in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                            try:
                                with open(path, 'rb') as f:
                                    file_info['data'] = base64.b64encode(f.read()).decode('utf-8')
                                file_info['type'] = 'pdf' if file_info['extension'] == '.pdf' else 'image'
                            except Exception:
                                continue
                        
                        files.append(file_info)
                
                return files if files else None
                
        finally:
            win32clipboard.CloseClipboard()
            
    except ImportError:
        # pywin32 not installed
        pass
    except Exception:
        pass
    
    return None


def _get_mac_clipboard_files() -> Optional[List[Dict[str, Any]]]:
    """Get files from macOS clipboard using pasteboard."""
    try:
        import subprocess
        import plistlib
        
        # Use osascript to get file URLs from clipboard
        script = '''
        on run
            set fileList to {}
            try
                set clipboardItems to (the clipboard as «class furl»)
                if clipboardItems is not {} then
                    set fileList to {POSIX path of clipboardItems}
                else
                    set clipboardItems to (the clipboard as list of «class furl»)
                    repeat with i in clipboardItems
                        set end of fileList to POSIX path of i
                    end repeat
                end if
            end try
            return fileList
        end run
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse the AppleScript list output
            file_paths = result.stdout.strip()
            if file_paths.startswith('{') and file_paths.endswith('}'):
                file_paths = file_paths[1:-1]  # Remove braces
                
            files = []
            for file_path in file_paths.split(', '):
                file_path = file_path.strip().strip('"')
                if file_path:
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        file_info = {
                            'path': str(path),
                            'filename': path.name,
                            'size': path.stat().st_size,
                            'extension': path.suffix.lower()
                        }
                        
                        # Read file content for supported types
                        if file_info['extension'] in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                            try:
                                with open(path, 'rb') as f:
                                    file_info['data'] = base64.b64encode(f.read()).decode('utf-8')
                                file_info['type'] = 'pdf' if file_info['extension'] == '.pdf' else 'image'
                            except Exception:
                                continue
                        
                        files.append(file_info)
            
            return files if files else None
            
    except Exception:
        pass
    
    return None


def _get_linux_clipboard_files() -> Optional[List[Dict[str, Any]]]:
    """Get files from Linux clipboard using xclip."""
    try:
        import subprocess
        
        # Try to get file list from clipboard using xclip
        result = subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'text/uri-list', '-o'],
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            files = []
            for uri in result.stdout.strip().split('\n'):
                if uri.startswith('file://'):
                    file_path = uri[7:]  # Remove 'file://'
                    # URL decode the path
                    from urllib.parse import unquote
                    file_path = unquote(file_path)
                    
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        file_info = {
                            'path': str(path),
                            'filename': path.name,
                            'size': path.stat().st_size,
                            'extension': path.suffix.lower()
                        }
                        
                        # Read file content for supported types
                        if file_info['extension'] in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                            try:
                                with open(path, 'rb') as f:
                                    file_info['data'] = base64.b64encode(f.read()).decode('utf-8')
                                file_info['type'] = 'pdf' if file_info['extension'] == '.pdf' else 'image'
                            except Exception:
                                continue
                        
                        files.append(file_info)
            
            return files if files else None
            
    except Exception:
        pass
    
    return None


def install_clipboard_dependencies():
    """Provide instructions for installing platform-specific clipboard dependencies."""
    if sys.platform == 'win32':
        return "Install pywin32: pip install pywin32"
    elif sys.platform == 'darwin':
        return "macOS clipboard support uses built-in osascript"
    else:
        return "Install xclip: sudo apt-get install xclip (Debian/Ubuntu) or equivalent"