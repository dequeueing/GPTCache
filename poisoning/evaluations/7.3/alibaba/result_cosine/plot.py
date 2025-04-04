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
    if 'failed' not in json_file:
        continue

    with open(json_file, 'r') as f:  # Replace 'your_file.json' with your actual file name
        data = json.load(f)

    # Step 2: Extract question_adv_cosine_sim values
    cosine_sims = [item['question_adv_cosine_sim'] for item in data]

    # Step 3: Define bins for the histogram (e.g., 0 to 1 with 0.1 intervals)
    bins = np.arange(0, 1.1, 0.05)  # Bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    hist, bin_edges = np.histogram(cosine_sims, bins=bins)

    # Step 4: Print detailed data for each range
    print(f"Distribution of {json_file} question_adv_cosine_sim:")
    print("| Range         | Count |")
    print("|---------------|-------|")
    for i in range(len(hist)):
        print(f"| [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}) | {hist[i]}     |")

    # Step 5: Plot the histogram and save it
    plt.figure(figsize=(8, 6))
    plt.hist(cosine_sims, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Distribution of question_adv_cosine_sim')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xticks(bins)
    plt.savefig(f'{json_file}.png', dpi=300, bbox_inches='tight')  # Save as PNG
    plt.show()