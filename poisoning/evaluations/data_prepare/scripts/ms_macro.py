from datasets import load_dataset
import json
import random

dataset_id = 'microsoft/ms_marco'
file_name = dataset_id.split('/')[1] + '_targeted.json'
ds = load_dataset(dataset_id, 'v1.1', split="validation")

if __name__ == "__main__":
    # Convert dataset to list for easier manipulation
    data = list(ds)
    total_questions = len(data)
    print(f"Total questions in dataset: {total_questions}")
    print("First item example:", data[0])

    # Store results: {round_number: [(question, answer), ...]}
    results = {}
    used_indices = set()  # Track used questions to avoid repetition

    # Run 10 rounds
    for round_num in range(1, 11):
        round_questions = []
        # Pick 10 unique questions per round
        while len(round_questions) < 10:
            if len(used_indices) >= total_questions:
                raise ValueError("Not enough unique questions in dataset!")
            idx = random.randint(0, total_questions - 1)
            if idx not in used_indices:
                used_indices.add(idx)
                question = data[idx]['query']
                # MS MARCO has multiple answers in 'answers'; take the first one
                answer = data[idx]['answers'][0] if data[idx]['answers'] else "No answer provided"
                round_questions.append((question, answer))
        
        results[f"Round {round_num}"] = round_questions
        print(f"Round {round_num}: Selected {len(round_questions)} questions")

    # Print total unique questions selected
    print(f"Total unique questions selected: {len(used_indices)}")

    # Save results to JSON file
    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)
        print(f"Results saved to {file_name}")

    # Optional: Print a sample round
    print("\nSample output (Round 1):")
    for q, a in results["Round 1"]:
        print(f"Q: {q}\nA: {a}\n")