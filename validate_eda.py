
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def execute_notebook(notebook_filename):
    print(f"Reading notebook: {notebook_filename}...")
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Do not force kernel_name, let it pick up the environment or default
    ep = ExecutePreprocessor(timeout=600)
    
    try:
        print("Starting execution...")
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_filename)}})
        print(f"Successfully executed {notebook_filename}")
    except Exception as e:
        print(f"Error executing the notebook {notebook_filename}")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {e}")
        # Try to print cell index if available
        # Sometimes e has info about which cell failed
        sys.exit(1)

if __name__ == "__main__":
    execute_notebook(r"c:\Users\subam\Desktop\projects\Retail-Price-Optimization-MLOPS\notebooks\EDA.ipynb")
