#!/usr/bin/env python3
"""
Export marimo notebooks to Jupyter and HTML, then clean up all interactive code references
"""

import json
import re
import subprocess
from pathlib import Path

def clean_jupyter_notebook(notebook_path):
    """
    Clean a Jupyter notebook by removing/fixing interactive code references
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    cleaned_cells = []
    has_interactive_elements = False
    
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            cleaned_cells.append(cell)
            continue
        
        source_lines = cell.get('source', [])
        source = ''.join(source_lines)
        
        # Skip cells that are primarily marimo UI definitions, but NOT if they have hide_code metadata
        # Check if this is a hidden cell first
        cell_metadata = cell.get('metadata', {})
        is_hidden_cell = cell_metadata.get('tags', []) and 'hide-input' in cell_metadata.get('tags', [])
        
        # Check if this cell references weather_categories and needs the definition added
        needs_weather_categories = 'weather_categories' in source
        needs_y_inputs = 'y_inputs' in source and 'y_inputs =' not in source
        needs_h_inputs = 'h_inputs' in source and 'h_inputs =' not in source
        needs_test_accuracy = 'test_accuracy' in source and 'test_accuracy =' not in source
        needs_vertical_filter = 'vertical_filter' in source and 'vertical_filter =' not in source
        # Shape UI elements are now filtered out entirely, so no need for fallback definitions
        
        skip_patterns = [
            r'^\s*[a-zA-Z_]\w*\s*=\s*mo\.ui\.',  # Variable assignments to mo.ui
            r'^\s*dropdown\s*=\s*mo\.ui\.dropdown',
            r'^\s*input_array\s*=\s*mo\.ui\.array',
            r'^\s*.*_weight_array\s*=\s*mo\.ui\.array',
            r'^\s*.*_slider\s*=\s*mo\.ui\.slider',
            r'^\s*.*_shape_ui\s*=\s*mo\.ui\.',  # Layer shape UI elements
            r'^\s*input_shape_ui\s*=\s*mo\.ui\.',  # Input shape UI elements
        ]
        
        # Check if this cell should be skipped entirely (but preserve hidden cells)
        should_skip = False
        if not is_hidden_cell:  # Only skip if it's not a hidden cell
            for pattern in skip_patterns:
                if re.search(pattern, source, re.MULTILINE):
                    should_skip = True
                    has_interactive_elements = True
                    break
        
        if should_skip:
            continue
        
        # Clean up references to interactive variables
        # Replace undefined interactive variables with sensible defaults
        cleaned_source = source
        
        # If this cell needs weather_categories, prepend the definition
        if needs_weather_categories:
            cleaned_source = "weather_categories = ['sunny', 'rainy', 'cloudy', 'snowy']\n" + cleaned_source
            
        # If this cell needs y_inputs, prepend the definition
        if needs_y_inputs:
            cleaned_source = "y_inputs = [-1.6, 0.8, -2.3]\n" + cleaned_source  # 3 logit values for 3 outputs
            
        # If this cell needs h_inputs, prepend the definition  
        if needs_h_inputs:
            cleaned_source = "h_inputs = [1.73, -1.19]\n" + cleaned_source  # Values that give sigmoid outputs of ~0.85, ~0.23
            
        # If this cell needs test_accuracy, prepend the definition
        if needs_test_accuracy:
            cleaned_source = "test_accuracy = 65.4  # Realistic test accuracy for a simple FCNN on CIFAR-10\n" + cleaned_source
            
        # If this cell needs vertical_filter, prepend the definition
        if needs_vertical_filter:
            cleaned_source = """import torch
vertical_filter = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
horizontal_filter = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
""" + cleaned_source
            
        # Shape UI elements are filtered out, so no fallback definitions needed
        
        # Remove marimo import statements
        cleaned_source = re.sub(r'^\s*import\s+marimo\s+as\s+mo\s*\n?', '', cleaned_source, flags=re.MULTILINE)
        cleaned_source = re.sub(r'^\s*import\s+marimo\s*\n?', '', cleaned_source, flags=re.MULTILINE)
        
        # Replace common interactive variable references
        replacements = {
            # UI array references
            r'h1_weight_array\[\d+\]\.value': '0.0',
            r'h2_weight_array\[\d+\]\.value': '0.0', 
            r'y\d+_weight_array\[\d+\]\.value': '0.0',
            r'dropdown\.value': '0',
            r'input_array\[\d+\]\.value': '1.0',
            r'label_dropdown\.value': '0',
            
            # General UI element .value references
            r'(\w+_input_ui)\.value': r'"example_text"',  # Replace UI input values with example text
            r'(\w+_ui)\.value': r'"default"',  # General UI value references
            
            # Keep input_shape_ui references as-is since we define MockUIValue class
            
            # Replace variable references with array indexing (common usage pattern)
            r'h_values\[': '[0.85, 0.23][',  # Realistic sigmoid outputs (0-1 range)
            r'var_values\[': '[1, 0.72, 0.34, 0.89][',  # Realistic normalized input values (0-1 range)
            r'weight_values\[': '[1.0, 1.0, 1.0, 1.0][',
            r'y_values\[': '[0.2, 0.7, 0.1][',  # 3 probabilities: 0.2 + 0.7 + 0.1 = 1.0
            
            # Convert mo.md() to print() to display content, remove other mo.* functions
            r'mo\.md\(': 'print(',
            r'mo\.(?!md)\w+\([^)]*\)': '# Interactive UI elements removed for static notebook',
            
            # Remove lines that are just mo.* calls without assignment (except mo.md)
            r'^\s*mo\.(?!md)\w+.*$': '',
        }
        
        for pattern, replacement in replacements.items():
            original_source = cleaned_source
            cleaned_source = re.sub(pattern, replacement, cleaned_source)
            if original_source != cleaned_source:
                has_interactive_elements = True
        
        # Simplify diagram text to remove complex calculations and show only final probabilities
        # Remove specific diagram calculation text patterns
        diagram_simplifications = {
            # Replace specific variable values in node labels with generic ones
            r'\{var_values\[\d+\]\}': 'val',
            r'\{h_inputs\[\d+\]\}': 'input',
            r'\{h_values\[\d+\]\}': 'val',
            r'\{y_values\[\d+\]\}': 'val',
            r'\{w\[\d+,\d+\]\}': 'w',
            r'\{v\[\d+,\s*\d+\]\}': 'w',
            
            # Show realistic loss calculation
            r'f"Loss: -log\([^)]+\) = \{[^}]+\}"': '"Loss: -log(0.7) = 0.357"',
            # Simplify softmax calculation text  
            r'f"softmax\([^)]+\)"': '"softmax(logits)"',
            # Replace final output displays with normalized probabilities (3 values that sum to 1.0)
            r'f"\{[^}]*y_values[^}]*\}"': '"[0.2, 0.7, 0.1]"',
        }
        
        for pattern, replacement in diagram_simplifications.items():
            cleaned_source = re.sub(pattern, replacement, cleaned_source)
        
        # Remove lines that only contain variable references to UI elements
        # But be careful not to break multi-line strings
        lines = cleaned_source.split('\n')
        filtered_lines = []
        in_multiline_string = False
        string_delimiter = None
        
        for line in lines:
            # Check for multiline string delimiters - handle multiple occurrences on same line
            triple_double_count = line.count('"""')
            triple_single_count = line.count("'''")
            
            # Handle triple double quotes
            if triple_double_count > 0:
                if not in_multiline_string:
                    if triple_double_count % 2 == 1:  # Odd number means start of multiline string
                        in_multiline_string = True
                        string_delimiter = '"""'
                elif string_delimiter == '"""':
                    if triple_double_count % 2 == 1:  # Odd number means end of multiline string
                        in_multiline_string = False
                        string_delimiter = None
            
            # Handle triple single quotes
            elif triple_single_count > 0:
                if not in_multiline_string:
                    if triple_single_count % 2 == 1:  # Odd number means start of multiline string
                        in_multiline_string = True
                        string_delimiter = "'''"
                elif string_delimiter == "'''":
                    if triple_single_count % 2 == 1:  # Odd number means end of multiline string
                        in_multiline_string = False
                        string_delimiter = None
            
            # Don't filter lines inside multiline strings
            if in_multiline_string:
                filtered_lines.append(line)
                continue
                
            # Skip lines that are just UI variable references or marimo calls
            ui_only_patterns = [
                r'^\s*[a-zA-Z_]\w*_array\s*$',
                r'^\s*dropdown\s*$',
                r'^\s*input_array\s*$',
                r'^\s*label_dropdown\s*$',
                r'^\s*mo\.(?!md)\w+.*$',  # Any line starting with mo. except mo.md
                r'^\s*#\s*Interactive UI elements removed.*$',  # Remove our placeholder comments
                r'^\s*var_values\s*=\s*\[.*\].*$',  # Remove var_values array definitions
                r'^\s*(var_values|h_values|weight_values|y_values)\s*=\s*\[.*\].*$',  # Remove specific *_values array definitions only
                r'^\s*.*_shape_ui.*$',  # Remove any lines referencing shape UI elements
                r'^\s*input_shape_ui.*$',  # Remove input shape UI references
                r'^\s*_C\s*=.*$',  # Remove _C variable assignments
                r'^\s*_H\s*=.*$',  # Remove _H variable assignments  
                r'^\s*_W\s*=.*$',  # Remove _W variable assignments
            ]
            
            skip_line = False
            for pattern in ui_only_patterns:
                if re.match(pattern, line):
                    skip_line = True
                    has_interactive_elements = True
                    break
            
            if not skip_line:
                filtered_lines.append(line)
        
        cleaned_source = '\n'.join(filtered_lines)
        
        # Skip cells that became empty or only contain comments (but preserve function definitions and important constants)
        # Also skip cells that reference UI-derived variables
        important_patterns = ['def ', 'NUM_CLASSES', 'EPOCHS', 'class ', 'test_accuracy', 'model.eval', 'torch.no_grad', 'vertical_filter', 'horizontal_filter', 'torch.tensor']
        has_important_content = any(pattern in cleaned_source for pattern in important_patterns)
        
        # Skip cells that reference UI-derived dimension variables
        ui_derived_patterns = ['_C', '_H', '_W', 'input_shape =', 'layer_shape =']
        has_ui_derived_content = any(pattern in cleaned_source for pattern in ui_derived_patterns)
        
        if (not cleaned_source.strip() or 
            (cleaned_source.strip().startswith('#') and not has_important_content) or
            has_ui_derived_content):
            continue
        
        # Update cell source - ensure proper line formatting
        lines = cleaned_source.split('\n')
        # Add newline to each line except the last one (which gets it conditionally)
        cell['source'] = [line + '\n' for line in lines[:-1]]
        if lines:
            # Handle the last line
            if lines[-1].strip():  # If last line has content
                cell['source'].append(lines[-1] + '\n')
            elif len(lines) > 1:  # If last line is empty and we have other lines
                cell['source'].append('\n')
        
        # Preserve hidden cell metadata if it was originally hidden
        if is_hidden_cell:
            if 'metadata' not in cell:
                cell['metadata'] = {}
            if 'tags' not in cell['metadata']:
                cell['metadata']['tags'] = []
            if 'hide-input' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide-input')
        
        cleaned_cells.append(cell)
    
    # Add warning cell at the beginning only if interactive elements were found
    if has_interactive_elements:
        warning_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "⚠️ **Static Version Notice**\n",
                "\n",
                "This is a static export of an interactive marimo notebook. Some features have been modified for compatibility:\n",
                "\n",
                "- Interactive UI elements (sliders, dropdowns, text inputs) have been removed\n",
                "- UI variable references have been replaced with default values\n",
                "- Some cells may have been simplified or removed entirely\n",
                "\n",
                "For the full interactive experience, please run the original marimo notebook (.py file) using:\n",
                "```bash\n",
                "uv run marimo edit notebook_name.py\n",
                "```\n",
                "\n",
                "---\n"
            ]
        }
        
        # Insert warning at the beginning
        cleaned_cells.insert(0, warning_cell)
    
    notebook['cells'] = cleaned_cells
    
    # Write cleaned notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

def export_and_clean_notebooks():
    """
    Export all marimo notebooks and clean up interactive code
    """
    notebooks_dir = Path('notebooks')
    jupyter_dir = Path('jupyter_notebooks')
    html_dir = Path('html_notebooks')
    
    # Create output directories
    jupyter_dir.mkdir(exist_ok=True)
    html_dir.mkdir(exist_ok=True)
    
    print("Exporting and cleaning notebooks...")
    
    for notebook_path in notebooks_dir.glob('*.py'):
        print(f"Processing {notebook_path.name}...")
        
        try:
            # Step 1: Export to Jupyter using marimo
            jupyter_output = jupyter_dir / f"{notebook_path.stem}.ipynb"
            jupyter_result = subprocess.run([
                'marimo', 'export', 'ipynb', str(notebook_path),
                '-o', str(jupyter_output), '-f'
            ], capture_output=True, text=True)
            
            if jupyter_result.returncode != 0:
                print(f"  ✗ Jupyter export failed: {jupyter_result.stderr}")
                continue
            
            print(f"  ✓ Jupyter export successful")
            
            # Step 2: Clean the Jupyter notebook
            clean_jupyter_notebook(jupyter_output)
            print(f"  ✓ Interactive code cleaned")
            
            # Step 3: Export cleaned notebook to HTML
            html_result = subprocess.run([
                'jupyter', 'nbconvert', '--to', 'html',
                str(jupyter_output), '--output-dir', str(html_dir)
            ], capture_output=True, text=True)
            
            if html_result.returncode == 0:
                print(f"  ✓ HTML export successful")
            else:
                print(f"  ⚠ HTML export warning: {html_result.stderr}")
                
        except Exception as e:
            print(f"  ✗ Error processing {notebook_path.name}: {e}")
    
    print("\nExport and cleaning complete!")
    print(f"Jupyter notebooks: {jupyter_dir}")
    print(f"HTML notebooks: {html_dir}")
    print("\nCleaning summary:")
    print("- Removed marimo UI element definitions (sliders, dropdowns, etc.)")
    print("- Replaced interactive variable references with default values")
    print("- Removed empty cells and UI-only display code")
    print("- Preserved all educational content and code examples")

if __name__ == '__main__':
    export_and_clean_notebooks()