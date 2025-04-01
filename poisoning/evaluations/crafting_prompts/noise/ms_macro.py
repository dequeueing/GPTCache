from datasets import load_dataset
import json
import random

dataset_id = 'microsoft/ms_marco'
file_name = dataset_id.split('/')[1] + '_noise.json'
ds = load_dataset(dataset_id, 'v1.1', split="validation")

if __name__ == "__main__":
    # Convert dataset to list for easier manipulation
    data = list(ds)
    total_questions = len(data)
    print(f"Total questions in dataset: {total_questions}")
    print("First item example:", data[0])

    # Store results: List of (question, answer) pairs
    results = []
    used_indices = set()  # Track used questions to avoid repetition

    # Number of unique questions to select
    num_questions_to_select = 2500

    # Check if there are enough questions in the dataset
    if num_questions_to_select > total_questions:
        raise ValueError("Not enough unique questions in dataset!")

    # Select num_questions_to_select unique questions
    while len(results) < num_questions_to_select:
        idx = random.randint(0, total_questions - 1)
        if idx not in used_indices:
            used_indices.add(idx)
            question = data[idx]['query']
            # MS MARCO has multiple answers in 'answers'; take the first one
            results.append(question)

    # Save results to JSON file
    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)
        print(f"Results saved to {file_name}")

