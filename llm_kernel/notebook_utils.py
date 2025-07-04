"""
Notebook utilities for accessing notebook file and cells directly.

This module provides utilities to find and read the current notebook file,
allowing access to all cells regardless of execution status.
"""

import os
import re
import json
import requests
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import ipykernel
except ImportError:
    ipykernel = None

try:
    from notebook.notebookapp import list_running_servers
except ImportError:
    try:
        from jupyter_server.serverapp import list_running_servers
    except ImportError:
        list_running_servers = None


class NotebookUtils:
    """Utilities for working with Jupyter notebooks from within a kernel."""
    
    def __init__(self, kernel_instance=None):
        self.kernel = kernel_instance
        self.log = kernel_instance.log if kernel_instance else None
        self._notebook_path = None
        self._notebook_data = None
    
    def get_notebook_path(self) -> Optional[str]:
        """
        Get the path to the current notebook file.
        
        Returns:
            Path to the notebook file, or None if not found
        """
        if self._notebook_path:
            return self._notebook_path
            
        # Try different methods to get the notebook path
        path = self._get_notebook_path_from_api()
        if not path:
            path = self._get_notebook_path_from_env()
        if not path:
            path = self._get_notebook_path_heuristic()
            
        if path and os.path.exists(path):
            self._notebook_path = path
            if self.log:
                self.log.info(f"Found notebook at: {path}")
        
        return self._notebook_path
    
    def _get_notebook_path_from_api(self) -> Optional[str]:
        """Get notebook path using Jupyter API."""
        if not ipykernel or not list_running_servers:
            return None
            
        try:
            # Get kernel ID from connection file
            connection_file = ipykernel.get_connection_file()
            kernel_id_match = re.search(r'kernel-(.*?)\.json', connection_file)
            if not kernel_id_match:
                return None
            kernel_id = kernel_id_match.group(1)
            
            # Find the notebook using the kernel ID
            for server in list_running_servers():
                try:
                    url = server['url']
                    token = server.get('token', '')
                    
                    # Get sessions
                    if token:
                        response = requests.get(f"{url}api/sessions", 
                                              params={'token': token})
                    else:
                        response = requests.get(f"{url}api/sessions")
                    
                    for session in response.json():
                        if session['kernel']['id'] == kernel_id:
                            notebook_path = os.path.join(
                                server['notebook_dir'],
                                session['notebook']['path']
                            )
                            return notebook_path
                except Exception:
                    continue
                    
        except Exception as e:
            if self.log:
                self.log.debug(f"API method failed: {e}")
        
        return None
    
    def _get_notebook_path_from_env(self) -> Optional[str]:
        """Try to get notebook path from environment variables."""
        # Check common environment variables
        for env_var in ['JPY_SESSION_NAME', 'NOTEBOOK_PATH', 'JUPYTER_PATH']:
            path = os.environ.get(env_var)
            if path and path.endswith('.ipynb') and os.path.exists(path):
                return path
        
        return None
    
    def _get_notebook_path_heuristic(self) -> Optional[str]:
        """Use heuristics to find the notebook file."""
        # Look for .ipynb files in current directory
        cwd = os.getcwd()
        ipynb_files = list(Path(cwd).glob('*.ipynb'))
        
        # If there's only one notebook, assume it's the one
        if len(ipynb_files) == 1:
            return str(ipynb_files[0])
        
        # Try to match based on kernel info if available
        if self.kernel and hasattr(self.kernel, 'session'):
            session_name = getattr(self.kernel.session, 'session', None)
            if session_name:
                for ipynb in ipynb_files:
                    if session_name in str(ipynb):
                        return str(ipynb)
        
        return None
    
    def read_notebook(self, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        Read the notebook file and return its contents.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Notebook data as a dictionary, or None if not found
        """
        if self._notebook_data and not force_reload:
            return self._notebook_data
            
        path = self.get_notebook_path()
        if not path:
            return None
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._notebook_data = json.load(f)
            return self._notebook_data
        except Exception as e:
            if self.log:
                self.log.error(f"Failed to read notebook: {e}")
            return None
    
    def get_all_cells(self) -> List[Dict[str, Any]]:
        """
        Get all cells from the notebook file.
        
        Returns:
            List of cell dictionaries
        """
        notebook = self.read_notebook()
        if not notebook:
            return []
            
        return notebook.get('cells', [])
    
    def get_cells_as_context(self, 
                           include_outputs: bool = True,
                           skip_empty: bool = True,
                           max_cells: Optional[int] = None,
                           up_to_cell: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get notebook cells formatted as context messages for LLM.
        
        Args:
            include_outputs: Include cell outputs
            skip_empty: Skip empty cells
            max_cells: Maximum number of cells to include
            up_to_cell: Only include cells up to this index (exclusive)
            
        Returns:
            List of messages in LLM format
        """
        cells = self.get_all_cells()
        messages = []
        
        for i, cell in enumerate(cells):
            # Stop at up_to_cell if specified
            if up_to_cell is not None and i >= up_to_cell:
                break
                
            if max_cells and i >= max_cells:
                break
                
            cell_type = cell.get('cell_type', 'code')
            source = ''.join(cell.get('source', []))
            
            if skip_empty and not source.strip():
                continue
                
            # Check if any line in the cell is a magic command (to include it)
            has_magic = any(line.strip().startswith('%') for line in source.splitlines())
            
            # Skip cells that are ONLY magic commands (except %%llm variants)
            if has_magic and not any(line.strip() and not line.strip().startswith('%') 
                                   for line in source.splitlines()):
                if not source.strip().startswith('%%llm'):
                    continue
                
            # Skip hide magic
            if source.strip().startswith('%%hide'):
                continue
                
            # Add cell content as user message
            if cell_type == 'markdown':
                messages.append({"role": "user", "content": f"[Markdown]: {source.strip()}"})
            else:
                # Handle %%llm cell magics
                if source.startswith('%%llm'):
                    lines = source.split('\n', 1)
                    if len(lines) > 1:
                        source = lines[1]
                    else:
                        continue
                        
                messages.append({"role": "user", "content": source.strip()})
                
                # Add output if available and requested
                if include_outputs and 'outputs' in cell:
                    output_text = []
                    for output in cell['outputs']:
                        if 'text' in output:
                            output_text.append(''.join(output['text']))
                        elif 'data' in output and 'text/plain' in output['data']:
                            output_text.append(''.join(output['data']['text/plain']))
                    
                    if output_text:
                        combined_output = '\n'.join(output_text).strip()
                        if combined_output:
                            messages.append({"role": "assistant", "content": combined_output})
        
        return messages
    
    def get_cell_at_position(self, position: int) -> Optional[Dict[str, Any]]:
        """Get a specific cell by position."""
        cells = self.get_all_cells()
        if 0 <= position < len(cells):
            return cells[position]
        return None
    
    def count_cells(self) -> int:
        """Count total number of cells in the notebook."""
        return len(self.get_all_cells())
    
    def find_cell_index(self, cell_content: str) -> Optional[int]:
        """
        Find the index of a cell by its content.
        
        Args:
            cell_content: The content to search for
            
        Returns:
            Cell index if found, None otherwise
        """
        cells = self.get_all_cells()
        
        # Normalize the content for comparison
        search_content = cell_content.strip()
        
        for i, cell in enumerate(cells):
            cell_source = ''.join(cell.get('source', [])).strip()
            if cell_source == search_content:
                return i
                
        return None