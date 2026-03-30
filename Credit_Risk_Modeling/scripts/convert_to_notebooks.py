"""
Script: convert_to_notebooks.py
Purpose: Convert all cell-structured .py scripts to .ipynb notebooks
         using jupytext. Run this once before sharing notebooks with
         stakeholders or uploading to GitHub notebooks/ folder.
Usage:   python scripts/convert_to_notebooks.py
"""

import subprocess
from pathlib import Path

scripts_dir   = Path("scripts")
notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

# All numbered pipeline scripts
pipeline_scripts = sorted([
    f for f in scripts_dir.glob("[0-9]*.py")
    if f.name != "convert_to_notebooks.py"
])

print(f"Converting {len(pipeline_scripts)} scripts to notebooks...")
print(f"Output folder: {notebooks_dir}/\n")

for script in pipeline_scripts:
    out_name = script.stem + ".ipynb"
    out_path = notebooks_dir / out_name
    cmd = ["jupytext", "--to", "notebook", str(script), "--output", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  OK  {script.name} -> notebooks/{out_name}")
    else:
        print(f"  FAIL {script.name}: {result.stderr.strip()}")

print(f"\nDone. Notebooks saved to: {notebooks_dir}/")
print("Open any .ipynb in Jupyter or upload to GitHub.")
