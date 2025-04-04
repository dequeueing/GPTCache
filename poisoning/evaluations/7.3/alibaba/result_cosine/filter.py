import json
import os

# Get current directory
current_dir = os.getcwd()
json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]

# Process each JSON file separately
for json_file in json_files:
    file_path = os.path.join(current_dir, json_file)
    failed_attacks = []  # Reset for each file
    
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            if data.get("attack success") is False:
                failed_attacks.append(data)
        elif isinstance(data, list):
            for item in data:
                if item.get("attack success") is False:
                    failed_attacks.append(item)
    
    # Save to a new JSON file if there are failed attacks
    if failed_attacks:
        output_file = f"failed_{json_file}"
        with open(output_file, 'w') as f:
            json.dump(failed_attacks, f, indent=4)
        print(f"Saved {len(failed_attacks)} items with failed attacks from {json_file} to {output_file}")
    else:
        print(f"No failed attacks found in {json_file}")

print("Processing complete.")