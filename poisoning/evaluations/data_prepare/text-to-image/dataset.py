import pandas as pd
import json
import random


def convert_csv():
    # Load CSV file
    csv_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/results.csv'
    df = pd.read_csv(csv_path, sep='|')
    df.columns = df.columns.str.strip()  # Clean column names

    # Group captions by image_name
    grouped = df.groupby('image_name')['comment'].apply(list).reset_index()

    # Convert to JSON structure
    data = {"images": []}
    for _, row in grouped.iterrows():
        data["images"].append({
            "image_name": row['image_name'],
            "captions": row['comment']
        })

    # Save to JSON file
    with open('flickr30k.json', 'w') as f:
        json.dump(data, f, indent=2)



if __name__ == '__main__':
    # Load JSON file
    json_file = 'datasets_raw/flickr30k.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get list of items (images with captions)
    items = data
    total_items = len(items)
    print(f"Total items: {total_items}")

    # Sample 100 items over 10 rounds (10 per round, with replacement)
    all_samples = []
    for _ in range(10):
        round_samples = random.choices(items, k=10)
        all_samples.extend(round_samples)

    # Save sampled data to a new JSON file
    output_file = 'datasets/target_questions.json'
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"Saved 100 samples to {output_file}")