import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the JSON file
# Get current directory
current_dir = os.getcwd()
json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]

# Process each JSON file separately
for json_file in json_files:
    with open(json_file, 'r') as f:  # Replace 'your_file.json' with your actual file name
        data = json.load(f)
    
    # Step 2: Filter items where injection_success is false
    failed_injections = [item for item in data if item.get("injection_success") is False]

    # Step 3: Save filtered items to a new JSON file
    output_file = 'failed_' + json_file
    with open(output_file, 'w') as f:
        json.dump(failed_injections, f, indent=4)

    # Step 4: Print result
    print(f"Saved {len(failed_injections)} items with failed injections to {output_file}")
