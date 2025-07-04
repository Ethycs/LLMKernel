#!/usr/bin/env python3
"""Clear all outputs from a Jupyter notebook."""

import json
import sys

def clear_outputs(notebook_path):
    """Clear all outputs from a notebook."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Clear outputs from all cells
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    
    # Write back
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Cleared outputs from {notebook_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clear_notebook_outputs.py notebook.ipynb")
        sys.exit(1)
    
    clear_outputs(sys.argv[1])