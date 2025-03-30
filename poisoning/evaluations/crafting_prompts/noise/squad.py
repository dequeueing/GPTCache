from datasets import load_dataset
import json
import random

dataset_id = 'rajpurkar/squad'
file_name = dataset_id.split('/')[1] + '_noise.json'
ds = load_dataset(dataset_id, split="validation")

if __name__ == "__main__":
    # Convert dataset to list for easier manipulation
    data = list(ds)
    total_questions = len(data)
    print(f"Total questions in dataset: {total_questions}")

    # Store results: {round_number: [(question, answer), ...]}
    noise = []
    used_indices = set()  # Track used questions to avoid repetition

    # Number of unique questions to select
    num_questions_to_select = 1500

    # Check if there are enough questions
    if num_questions_to_select > total_questions:
        raise ValueError("Not enough unique questions in dataset!")

    # Select 1500 unique questions
    while len(noise) < num_questions_to_select:
        idx = random.randint(0, total_questions - 1)
        if idx not in used_indices:
            used_indices.add(idx)
            question = data[idx]['question']
            # SQuAD has multiple answers; take the first one for simplicity
            noise.append((question))

    # Save results to JSON file
    with open(file_name, "w") as file:
        json.dump(noise, file, indent=4)
        print(f"Results saved to {file_name}")