#!/usr/bin/env python3
"""
Methods to get the current notebook file path from within a Jupyter kernel.

This file demonstrates various approaches to accessing the notebook filename
and path from within the kernel execution context.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any


# Method 1: Using ipykernel connection file and Jupyter server API
def get_notebook_path_via_api() -> Optional[str]:
    """
    Get the notebook path by querying the Jupyter server API.
    
    This method:
    1. Gets the kernel ID from the connection file
    2. Queries all running Jupyter servers
    3. Finds the session with matching kernel ID
    4. Returns the notebook path
    
    Returns:
        str: Full path to the notebook file, or None if not found
    """
    try:
        import ipykernel
        import requests
        from requests.compat import urljoin
        
        # Try to import from notebook first (Jupyter Notebook)
        try:
            from notebook.notebookapp import list_running_servers
        except ImportError:
            # Fall back to jupyter_server (JupyterLab)
            try:
                from jupyter_server.serverapp import list_running_servers
            except ImportError:
                logging.warning("Neither notebook nor jupyter_server available")
                return None
        
        # Get kernel ID from connection file
        connection_file = ipykernel.get_connection_file()
        kernel_id_match = re.search(r'kernel-(.*?)\.json', connection_file)
        
        if not kernel_id_match:
            logging.warning("Could not extract kernel ID from connection file")
            return None
            
        kernel_id = kernel_id_match.group(1)
        
        # Check all running servers
        for server in list_running_servers():
            try:
                response = requests.get(
                    urljoin(server['url'], 'api/sessions'),
                    params={'token': server.get('token', '')},
                    timeout=5
                )
                
                for session in response.json():
                    if session['kernel']['id'] == kernel_id:
                        notebook_path = session['notebook']['path']
                        full_path = os.path.join(server['notebook_dir'], notebook_path)
                        return full_path
                        
            except Exception as e:
                logging.debug(f"Error querying server {server['url']}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Error getting notebook path via API: {e}")
        
    return None


# Method 2: Using IPython's global state (less reliable)
def get_notebook_path_via_ipython() -> Optional[str]:
    """
    Try to get notebook path from IPython's global state.
    
    This method is less reliable and may not work in all environments.
    
    Returns:
        str: Path to the notebook file, or None if not found
    """
    try:
        from IPython import get_ipython
        
        ipython = get_ipython()
        if not ipython:
            return None
            
        # Check if we're in a notebook environment
        if not hasattr(ipython, 'kernel'):
            return None
            
        # Try to access notebook document (works in some environments)
        if hasattr(ipython, 'notebook_dir'):
            return ipython.notebook_dir
            
        # Try to get from config
        if hasattr(ipython, 'config'):
            notebook_dir = ipython.config.get('NotebookApp', {}).get('notebook_dir')
            if notebook_dir:
                return notebook_dir
                
    except Exception as e:
        logging.debug(f"Error getting notebook path via IPython: {e}")
        
    return None


# Method 3: Using environment variables and working directory
def get_notebook_path_heuristic() -> Optional[str]:
    """
    Use heuristics to guess the notebook path.
    
    This method checks:
    1. JUPYTER_PATH environment variable
    2. Current working directory for .ipynb files
    3. Parent directories for .ipynb files
    
    Returns:
        str: Likely path to a notebook file, or None if not found
    """
    try:
        # Check JUPYTER_PATH
        jupyter_path = os.environ.get('JUPYTER_PATH')
        if jupyter_path:
            return jupyter_path
            
        # Check current directory for notebook files
        cwd = Path.cwd()
        notebooks = list(cwd.glob('*.ipynb'))
        
        # If there's only one notebook, assume it's the current one
        if len(notebooks) == 1:
            return str(notebooks[0])
            
        # Check parent directories
        for parent in cwd.parents:
            notebooks = list(parent.glob('*.ipynb'))
            if len(notebooks) == 1:
                return str(notebooks[0])
                
    except Exception as e:
        logging.debug(f"Error in heuristic notebook path detection: {e}")
        
    return None


# Method 4: Using kernel connection info
def get_notebook_info_from_kernel(kernel_instance) -> Dict[str, Any]:
    """
    Extract notebook-related information from kernel instance.
    
    Args:
        kernel_instance: The kernel instance (self in kernel methods)
        
    Returns:
        dict: Information about the notebook environment
    """
    info = {
        'kernel_id': None,
        'connection_file': None,
        'session_id': None,
        'working_directory': str(Path.cwd()),
        'notebook_path': None
    }
    
    try:
        # Get connection file
        if hasattr(kernel_instance, 'connection_file'):
            info['connection_file'] = kernel_instance.connection_file
            
        # Get session info
        if hasattr(kernel_instance, 'session'):
            info['session_id'] = getattr(kernel_instance.session, 'session', None)
            
        # Get kernel ID
        if hasattr(kernel_instance, 'ident'):
            info['kernel_id'] = kernel_instance.ident.decode() if isinstance(kernel_instance.ident, bytes) else str(kernel_instance.ident)
            
        # Try to get notebook path using various methods
        notebook_path = get_notebook_path_via_api()
        if notebook_path:
            info['notebook_path'] = notebook_path
        else:
            # Fall back to heuristics
            notebook_path = get_notebook_path_heuristic()
            if notebook_path:
                info['notebook_path'] = notebook_path
                info['notebook_path_method'] = 'heuristic'
                
    except Exception as e:
        logging.error(f"Error extracting notebook info from kernel: {e}")
        
    return info


# Method 5: For use within a magic command or kernel method
def get_current_notebook_path(kernel_self) -> Optional[str]:
    """
    Convenience method to get current notebook path from within kernel.
    
    This should be called from within a kernel method where 'self' is available.
    
    Args:
        kernel_self: The kernel instance (self)
        
    Returns:
        str: Path to the current notebook, or None if not found
    """
    # First try the API method
    notebook_path = get_notebook_path_via_api()
    if notebook_path:
        return notebook_path
        
    # Try to get from kernel info
    info = get_notebook_info_from_kernel(kernel_self)
    if info.get('notebook_path'):
        return info['notebook_path']
        
    # Fall back to heuristics
    return get_notebook_path_heuristic()


# Example usage in a kernel method:
def example_kernel_method(self):
    """Example of how to use these methods within a kernel."""
    # Method 1: Direct API call
    notebook_path = get_notebook_path_via_api()
    if notebook_path:
        print(f"Notebook path (via API): {notebook_path}")
        
    # Method 2: Using kernel info
    info = get_notebook_info_from_kernel(self)
    print(f"Kernel info: {json.dumps(info, indent=2)}")
    
    # Method 3: Convenience method
    current_notebook = get_current_notebook_path(self)
    if current_notebook:
        print(f"Current notebook: {current_notebook}")
    else:
        print("Could not determine current notebook path")


if __name__ == "__main__":
    # Test the methods (when run as a script)
    print("Testing notebook path detection methods...")
    
    print("\n1. API method:")
    path = get_notebook_path_via_api()
    print(f"   Result: {path}")
    
    print("\n2. IPython method:")
    path = get_notebook_path_via_ipython()
    print(f"   Result: {path}")
    
    print("\n3. Heuristic method:")
    path = get_notebook_path_heuristic()
    print(f"   Result: {path}")