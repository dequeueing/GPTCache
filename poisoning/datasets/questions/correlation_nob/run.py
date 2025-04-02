import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
embedding_model_type = 'distilbert-base-uncased'
embedding_model = AutoModel.from_pretrained(embedding_model_type).to('cuda').eval()
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)

# Function for sentence embedding
def _sent_embed_from_hidden(hidden_states, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    ).to(hidden_states.device)
    sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embs

# Optimized cosine similarity function
def cosine_sim_batch(queries1, queries2):
    queries1_tokenized = embedding_tokenizer(queries1, return_tensors="pt", padding=True, truncation=True).to('cuda')
    queries2_tokenized = embedding_tokenizer(queries2, return_tensors="pt", padding=True, truncation=True).to('cuda')
    
    with torch.no_grad():
        q1_embedding = embedding_model(**queries1_tokenized).last_hidden_state
        q2_embedding = embedding_model(**queries2_tokenized).last_hidden_state

        q1_sent_emb = _sent_embed_from_hidden(q1_embedding, queries1_tokenized['attention_mask'])
        q2_sent_emb = _sent_embed_from_hidden(q2_embedding, queries2_tokenized['attention_mask'])
        
        return torch.nn.functional.cosine_similarity(q1_sent_emb, q2_sent_emb).cpu().numpy()

def get_normal_distribution(dist_mean: float, number: int, all_noise, seed: int = 20):
    """Return a subset of noise IDs where cosine similarity follows a normal distribution."""
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility
    cosine_similarities = np.array([noise['dist'] for noise in all_noise])
    std_dev = np.std(cosine_similarities) if np.std(cosine_similarities) > 0 else 1  # Avoid division by zero
    probabilities = np.exp(-0.5 * ((cosine_similarities - dist_mean) / std_dev) ** 2)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(all_noise), size=number, p=probabilities, replace=False)
    return [all_noise[i]['id'] for i in selected_indices]

def plot_distribution(all_noise, selected_ids, dataset_id, goal_sim):
    """
    Plot the distribution of cosine similarities for all noise samples and the selected subset,
    with a vertical line indicating the goal similarity.

    Args:
        all_noise (list of dict): Full dataset, each entry should have "id" and "dist".
        selected_ids (list): IDs of selected noise samples.
        dataset_id (str): Identifier for the dataset.
        goal_sim (float): Target similarity value to mark with a vertical line.
    """
    # Input validation
    if not all_noise or not isinstance(all_noise, list):
        raise ValueError("all_noise must be a non-empty list")
    if not all(isinstance(item, dict) and 'id' in item and 'dist' in item for item in all_noise):
        raise ValueError("Each item in all_noise must be a dict with 'id' and 'dist' keys")
    if not isinstance(selected_ids, list):
        raise ValueError("selected_ids must be a list")

    # Extract cosine similarity values
    all_dist_values = np.array([noise['dist'] for noise in all_noise])
    selected_dist_values = np.array([noise['dist'] for noise in all_noise if noise['id'] in selected_ids])

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot distributions
    sns.histplot(all_dist_values, bins=50, kde=True, color='blue', alpha=0.3, label="All Noise Questions")
    sns.histplot(selected_dist_values, bins=30, kde=True, color='red', alpha=0.7, label="Selected Noise Questions")
    
    # Add vertical line for goal similarity
    plt.axvline(x=goal_sim, color='green', linestyle='--', label=f'Goal Similarity ({goal_sim:.2f})')

    # Labels and legend
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cosine Similarities: All vs. Selected")
    plt.legend()
    
    # Save and show
    timestamp = time.time()
    plt.savefig(f"plot/{dataset_id}_{timestamp}.png")
    plt.show()


# File paths
input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/noise/'
prompt_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/'

datasets = [
    "squad",
    "MedQuad-MedicalQnADataset",
    "ms_marco",
]

# Run batch processing
if __name__ == '__main__':
    for dataset_id in datasets:
        input_file = input_path + f"{dataset_id}.json"
        prompt_file = prompt_path + f"{dataset_id}.json"
        
        # Load data
        with open(input_file, 'r') as file:
            all_noise = json.load(file)
        with open(prompt_file, 'r') as file:
            target = json.load(file)
            
        # for item in [target[-1]]:  
        for i in range(0, len(target), 30):
            item = target[i]
            target_question = item['question']
            adv_question = item['adv']
            goal_sim = cosine_sim_batch(target_question, adv_question)[0]
            
            # Extract noise questions and IDs
            noise_questions = [noise['question'] for noise in all_noise]
            noise_ids = [noise['id'] for noise in all_noise]

            # Compute cosine similarity in batches
            batch_size = 2048
            all_similarities = []

            for i in tqdm(range(0, len(noise_questions), batch_size)):
                batch_noise_questions = noise_questions[i:i + batch_size]
                batch_similarities = cosine_sim_batch(batch_noise_questions, [target_question] * len(batch_noise_questions))
                
                # Store computed similarities
                all_similarities.extend(batch_similarities)

            # Attach cosine similarity to each noise sample
            for i, noise in enumerate(all_noise):
                noise["dist"] = all_similarities[i]

            # Select noise samples using normal distribution
            selected_ids = get_normal_distribution(dist_mean=0.5, number=500, all_noise=all_noise)
            plot_distribution(all_noise, selected_ids, dataset_id, goal_sim)

