#!/usr/bin/env python3
"""
Filter out hidden cells from marimo notebooks before export
"""

import os
import re
import ast
import subprocess
from pathlib import Path

def is_hidden_cell(cell_code):
    """
    Check if a marimo cell is marked as hidden.
    Hidden cells typically have hide_code=True in mo.md or similar markers.
    """
    # Look for common hidden cell patterns
    hidden_patterns = [
        r'hide_code\s*=\s*True',
        r'mo\.md\s*\(\s*r?["\'].*["\']\s*,.*hide_code\s*=\s*True',
        r'#\s*HIDDEN',
        r'#\s*@hidden',
    ]
    
    for pattern in hidden_patterns:
        if re.search(pattern, cell_code, re.IGNORECASE):
            return True
    
    return False

def filter_marimo_notebook(notebook_path):
    """
    Parse a marimo notebook and filter out hidden cells
    """
    with open(notebook_path, 'r') as f:
        content = f.read()
    
    # Split by cell definitions (looking for def __() patterns)
    cell_pattern = r'(def __\w*\(\):\s*(?:(?!^def __|\Z).*\n?)*)'
    cells = re.findall(cell_pattern, content, re.MULTILINE)
    
    # Filter out hidden cells
    filtered_cells = []
    for cell in cells:
        if not is_hidden_cell(cell):
            filtered_cells.append(cell)
    
    # Reconstruct notebook with imports and app definition
    lines = content.split('\n')
    
    # Get imports and initial setup (before first cell definition)
    header_lines = []
    in_header = True
    
    for line in lines:
        if line.strip().startswith('def __') and '__' in line:
            in_header = False
            break
        if in_header:
            header_lines.append(line)
    
    # Get the app.run() or similar footer
    footer_lines = []
    found_app_run = False
    for line in reversed(lines):
        if 'app.run()' in line or 'if __name__ ==' in line:
            found_app_run = True
        if found_app_run:
            footer_lines.insert(0, line)
        elif found_app_run:
            break
    
    # Combine header, filtered cells, and footer
    filtered_content = '\n'.join(header_lines) + '\n\n'
    filtered_content += '\n\n'.join(filtered_cells)
    if footer_lines:
        filtered_content += '\n\n' + '\n'.join(footer_lines)
    
    return filtered_content

def main():
    notebooks_dir = Path('notebooks')
    temp_dir = Path('temp_filtered_notebooks')
    jupyter_dir = Path('jupyter_notebooks')
    html_dir = Path('html_notebooks')
    
    # Create temporary directory for filtered notebooks
    temp_dir.mkdir(exist_ok=True)
    
    print("Filtering marimo notebooks and re-exporting...")
    
    for notebook_path in notebooks_dir.glob('*.py'):
        print(f"Processing {notebook_path.name}...")
        
        try:
            # Filter the notebook
            filtered_content = filter_marimo_notebook(notebook_path)
            
            # Save filtered version
            temp_notebook = temp_dir / notebook_path.name
            with open(temp_notebook, 'w') as f:
                f.write(filtered_content)
            
            # Export to Jupyter
            jupyter_output = jupyter_dir / f"{notebook_path.stem}.ipynb"
            subprocess.run([
                'marimo', 'export', 'ipynb', str(temp_notebook),
                '-o', str(jupyter_output), '-f'
            ], check=True, capture_output=True)
            
            # Export to HTML
            html_output = html_dir / f"{notebook_path.stem}.html"
            subprocess.run([
                'jupyter', 'nbconvert', '--to', 'html',
                str(jupyter_output), '--output', str(html_output.resolve())
            ], check=True, capture_output=True)
            
            print(f"  ✓ Exported {notebook_path.name} (filtered)")
            
        except Exception as e:
            print(f"  ✗ Error processing {notebook_path.name}: {e}")
            # Fallback to original export
            jupyter_output = jupyter_dir / f"{notebook_path.stem}.ipynb"
            subprocess.run([
                'marimo', 'export', 'ipynb', str(notebook_path),
                '-o', str(jupyter_output), '-f'
            ], check=True, capture_output=True)
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print("Filtering and export complete!")

if __name__ == '__main__':
    main()