# inspect_npz.py (Updated Version)
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <path_to_npz_file>")
    sys.exit(1)

npz_file = sys.argv[1]

if not os.path.exists(npz_file):
    print(f"Error: File not found at '{npz_file}'")
    sys.exit(1)

print(f"--- Inspecting keys in: {os.path.basename(npz_file)} ---")

try:
    data = np.load(npz_file)
    keys = data.files

    print(f"Found {len(keys)} total parameter arrays.")
    
    # 打印前10个参数名
    print("\n--- First 10 parameter names ---")
    for key in keys[:10]:
        print(key)
        
    # 打印最后10个参数名
    print("\n--- Last 10 parameter names ---")
    for key in keys[-10:]:
        print(key)

except Exception as e:
    print(f"Error loading file: {e}")