import json
import os


current_dir = os.getcwd()
json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]

# Process each JSON file separately
for json_file in json_files:

    # Step 1: Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Step 2: Modify the "white" field by adding "ONLY" before "print"
    for item in data:
        white_text = item['white']
        # Replace "print" with "ONLY print" (case-sensitive)
        updated_white = white_text.replace("print", "ONLY print")
        item['white'] = updated_white

    # Step 3: Save the modified data to a new JSON file
    output_file = 'only_' + json_file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Modified data saved to {output_file}")