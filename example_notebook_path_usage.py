#!/usr/bin/env python3
"""
Example usage of notebook path detection in LLM Kernel.

This shows how to integrate notebook path detection into magic commands
and kernel functionality.
"""

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import display, HTML, Markdown
import json
from pathlib import Path


@magics_class
class NotebookInfoMagics(Magics):
    """Magic commands for notebook information."""
    
    def __init__(self, shell, kernel_instance):
        super().__init__(shell)
        self.kernel = kernel_instance
        
        # Initialize notebook utils if available
        try:
            from llm_kernel.notebook_utils import NotebookUtils
            self.notebook_utils = NotebookUtils(kernel_instance)
        except ImportError:
            self.notebook_utils = None
    
    @line_magic
    def notebook_info(self, line):
        """Show information about the current notebook."""
        if not self.notebook_utils:
            print("❌ NotebookUtils not available")
            return
            
        # Get notebook path
        notebook_path = self.notebook_utils.get_notebook_path()
        
        info = {
            "Notebook Path": notebook_path or "Not found",
            "Working Directory": str(Path.cwd()),
            "Kernel ID": getattr(self.kernel, 'ident', 'Unknown'),
        }
        
        # If we found the notebook, get more info
        if notebook_path:
            nb_path = Path(notebook_path)
            info.update({
                "Notebook Name": nb_path.name,
                "Notebook Directory": str(nb_path.parent),
                "Exists": nb_path.exists(),
            })
            
            # Try to get notebook metadata
            metadata = self.notebook_utils.get_notebook_metadata()
            if metadata:
                info["Kernel Spec"] = metadata.get('kernelspec', {}).get('display_name', 'Unknown')
                info["Language"] = metadata.get('language_info', {}).get('name', 'Unknown')
        
        # Display as nice HTML table
        html = """
        <div style="margin: 10px 0;">
            <h3>📓 Notebook Information</h3>
            <table style="border-collapse: collapse; margin: 10px 0;">
        """
        
        for key, value in info.items():
            html += f"""
                <tr>
                    <td style="padding: 5px 15px 5px 0; font-weight: bold;">{key}:</td>
                    <td style="padding: 5px;">{value}</td>
                </tr>
            """
            
        html += """
            </table>
        </div>
        """
        
        display(HTML(html))
    
    @line_magic
    def notebook_cells(self, line):
        """List cells in the current notebook."""
        if not self.notebook_utils:
            print("❌ NotebookUtils not available")
            return
            
        # Parse arguments
        args = line.strip().split()
        cell_type = args[0] if args else None
        
        if cell_type and cell_type not in ['code', 'markdown']:
            print("Usage: %notebook_cells [code|markdown]")
            return
            
        # Get cells
        cells = self.notebook_utils.get_notebook_cells(cell_type=cell_type)
        
        if not cells:
            print("No cells found or could not read notebook")
            return
            
        # Display cells
        print(f"📓 Found {len(cells)} {cell_type or 'total'} cells:\n")
        
        for i, cell in enumerate(cells):
            cell_type_icon = "💻" if cell['cell_type'] == 'code' else "📝"
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
                
            # Truncate long content
            preview = source[:100] + "..." if len(source) > 100 else source
            preview = preview.replace('\n', ' ')
            
            print(f"{cell_type_icon} Cell {i+1}: {preview}")
    
    @line_magic
    def llm_context_from_notebook(self, line):
        """
        Load context from the current notebook file instead of history.
        
        This reads the actual notebook file and uses its cells as context.
        """
        if not self.notebook_utils:
            print("❌ NotebookUtils not available")
            return
            
        # Get notebook cells
        cells = self.notebook_utils.get_notebook_cells(cell_type='code')
        
        if not cells:
            print("❌ No code cells found or could not read notebook")
            return
            
        # Convert to context messages
        messages = []
        for cell in cells:
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
                
            if source.strip():
                # Skip magic commands
                if not source.strip().startswith('%'):
                    messages.append({
                        'role': 'user',
                        'content': source.strip()
                    })
                    
                    # Add outputs if available
                    outputs = cell.get('outputs', [])
                    output_text = []
                    for output in outputs:
                        if 'text' in output:
                            output_text.append(output['text'])
                        elif 'data' in output and 'text/plain' in output['data']:
                            output_text.append(output['data']['text/plain'])
                            
                    if output_text:
                        messages.append({
                            'role': 'assistant',
                            'content': '\n'.join(output_text)
                        })
        
        # Store as saved context
        if hasattr(self.kernel, 'saved_context'):
            self.kernel.saved_context = messages
            print(f"✅ Loaded {len(messages)} messages from notebook file")
            print("   (This context will be used for future LLM queries)")
        else:
            print("❌ Could not save context to kernel")


# Integration example for LLMKernel
def add_notebook_path_to_kernel(kernel_class):
    """
    Decorator to add notebook path functionality to a kernel class.
    
    Usage:
        @add_notebook_path_to_kernel
        class LLMKernel(IPythonKernel):
            ...
    """
    original_init = kernel_class.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # Add notebook utils
        try:
            from llm_kernel.notebook_utils import NotebookUtils
            self.notebook_utils = NotebookUtils(self)
            
            # Try to get notebook path on initialization
            notebook_path = self.notebook_utils.get_notebook_path()
            if notebook_path:
                self.log.info(f"Detected notebook: {notebook_path}")
                
        except Exception as e:
            self.log.warning(f"Could not initialize notebook utils: {e}")
            self.notebook_utils = None
    
    kernel_class.__init__ = new_init
    return kernel_class


# Example of using in context building
def get_context_from_notebook_file(kernel_instance):
    """
    Alternative context building that reads from the notebook file
    instead of execution history.
    """
    if not hasattr(kernel_instance, 'notebook_utils') or not kernel_instance.notebook_utils:
        return []
        
    # Get code cells from notebook
    cells = kernel_instance.notebook_utils.get_notebook_cells(cell_type='code')
    
    messages = []
    for cell in cells:
        source = cell.get('source', '')
        if isinstance(source, list):
            source = ''.join(source)
            
        # Skip empty cells and magic commands
        if source.strip() and not source.strip().startswith('%'):
            messages.append({
                'role': 'user',
                'content': source.strip()
            })
            
    return messages


if __name__ == "__main__":
    # Example of standalone usage
    from llm_kernel.notebook_utils import NotebookUtils
    
    utils = NotebookUtils()
    notebook_path = utils.get_notebook_path()
    
    if notebook_path:
        print(f"Found notebook: {notebook_path}")
        
        # Read notebook
        notebook = utils.read_notebook()
        if notebook:
            print(f"Notebook has {len(notebook.get('cells', []))} cells")
            
            # Get code cells
            code_cells = utils.get_notebook_cells(cell_type='code')
            print(f"Found {len(code_cells)} code cells")
    else:
        print("Could not find notebook path")