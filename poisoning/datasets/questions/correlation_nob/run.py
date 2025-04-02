import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder

semantic_model_type = "cross-encoder/quora-distilroberta-base"
semantic_encoder = CrossEncoder(semantic_model_type)

def semantic_score(q1, q2_list):
    pairs = [(q1, q2) for q2 in q2_list]
    scores = semantic_encoder.predict(pairs, show_progress_bar=False)
    return scores  # Returns a numpy array of scores

# Load model and tokenizer
embedding_model_type = 'distilbert-base-uncased'
embedding_model = AutoModel.from_pretrained(embedding_model_type).to('cuda').eval()
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)

def _sent_embed_from_hidden(hidden_states, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    ).to(hidden_states.device)
    sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embs

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
    if seed is not None:
        np.random.seed(seed)
    cosine_similarities = np.array([noise['dist'] for noise in all_noise])
    std_dev = np.std(cosine_similarities) if np.std(cosine_similarities) > 0 else 1
    probabilities = np.exp(-0.5 * ((cosine_similarities - dist_mean) / std_dev) ** 2)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(all_noise), size=number, p=probabilities, replace=False)
    return [all_noise[i]['id'] for i in selected_indices]


def plot_distribution(all_noise, selected_ids, dataset_id, goal_sim, target_question, adv_question):
    if not all_noise or not isinstance(all_noise, list):
        raise ValueError("all_noise must be a non-empty list")
    if not all(isinstance(item, dict) and 'id' in item and 'dist' in item for item in all_noise):
        raise ValueError("Each item in all_noise must be a dict with 'id' and 'dist' keys")
    if not isinstance(selected_ids, list):
        raise ValueError("selected_ids must be a list")

    all_dist_values = np.array([noise['dist'] for noise in all_noise])
    selected_dist_values = np.array([noise['dist'] for noise in all_noise if noise['id'] in selected_ids])
    
    # Calculate semantic scores for selected noise vs. target_question (batch)
    selected_noise_questions = [noise['question'] for noise in all_noise if noise['id'] in selected_ids]
    semantic_scores_target = semantic_score(target_question, selected_noise_questions)
    filtered_semantic_scores_target = semantic_scores_target[semantic_scores_target >= 0.7]
    if len(filtered_semantic_scores_target) == 0:
        print("Warning: No semantic scores (target) >= 0.7 found.")
        filtered_semantic_scores_target = np.array([0.7])

    # Calculate semantic scores for all noise vs. adv_question (batch)
    all_noise_questions = [noise['question'] for noise in all_noise]
    semantic_scores_adv = semantic_score(adv_question, all_noise_questions)
    filtered_semantic_scores_adv = semantic_scores_adv[semantic_scores_adv >= 0.7]
    if len(filtered_semantic_scores_adv) == 0:
        print("Warning: No semantic scores (adv) >= 0.7 found.")
        filtered_semantic_scores_adv = np.array([0.7])

    # Plot cosine similarity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_dist_values, bins=50, kde=True, color='blue', alpha=0.3, label="All Noise Questions")
    sns.histplot(selected_dist_values, bins=30, kde=True, color='red', alpha=0.7, label="Selected Noise Questions")
    plt.axvline(x=goal_sim, color='green', linestyle='--', label=f'Goal Similarity ({goal_sim:.2f})')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cosine Similarities: All vs. Selected")
    plt.legend()
    plt.savefig(f"plot/{dataset_id}_{mean}_cosine.png")
    plt.show()

    # Plot filtered semantic score distribution (target, >= 0.7)
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_semantic_scores_target, bins=30, kde=True, color='purple', alpha=0.7, label="Selected Semantic Scores (>= 0.7)")
    plt.xlim(0.7, 1.0)
    plt.xlabel("Semantic Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Semantic Scores: Selected vs. Target (>= 0.7)")
    plt.legend()
    plt.savefig(f"plot/{dataset_id}_{mean}_semantic_target.png")
    plt.show()

    # Plot filtered semantic score distribution (adv, >= 0.7)
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_semantic_scores_adv, bins=50, kde=True, color='orange', alpha=0.7, label="All Noise Semantic Scores (>= 0.7)")
    plt.xlim(0.7, 1.0)
    plt.xlabel("Semantic Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Semantic Scores: All Noise vs. Adv (>= 0.7)")
    plt.legend()
    plt.savefig(f"plot/{dataset_id}_{mean}_semantic_adv.png")
    plt.show()

    # Print top-scoring noise questions
    print(f"\nTop 5 Noise Questions with Highest Semantic Scores vs. Target:")
    target_indices = np.argsort(semantic_scores_target)[-5:][::-1]  # Top 5 indices
    for idx in target_indices:
        score = semantic_scores_target[idx]
        question = selected_noise_questions[idx]
        print(f"Score: {score:.4f}, Question: {question}")

    print(f"\nTop 5 Noise Questions with Highest Semantic Scores vs. Adv:")
    adv_indices = np.argsort(semantic_scores_adv)[-5:][::-1]  # Top 5 indices
    for idx in adv_indices:
        score = semantic_scores_adv[idx]
        question = all_noise_questions[idx]
        print(f"Score: {score:.4f}, Question: {question}")

    # Print numerical details
    print(f"\nDataset: {dataset_id}")
    print(f"Goal Similarity: {goal_sim:.4f}")
    print(f"All Noise Cosine - Mean: {np.mean(all_dist_values):.4f}, Std: {np.std(all_dist_values):.4f}")
    print(f"Selected Cosine - Mean: {np.mean(selected_dist_values):.4f}, Std: {np.std(selected_dist_values):.4f}")
    print(f"Selected Cosine - Min: {np.min(selected_dist_values):.4f}, Max: {np.max(selected_dist_values):.4f}")
    print(f"Selected Semantic (>= 0.7) vs. Target - Mean: {np.mean(filtered_semantic_scores_target):.4f}, Std: {np.std(filtered_semantic_scores_target):.4f}")
    print(f"Selected Semantic (>= 0.7) vs. Target - Min: {np.min(filtered_semantic_scores_target):.4f}, Max: {np.max(filtered_semantic_scores_target):.4f}")
    print(f"Selected Semantic Count (>= 0.7) vs. Target: {len(filtered_semantic_scores_target)}")
    print(f"All Noise Semantic (>= 0.7) vs. Adv - Mean: {np.mean(filtered_semantic_scores_adv):.4f}, Std: {np.std(filtered_semantic_scores_adv):.4f}")
    print(f"All Noise Semantic (>= 0.7) vs. Adv - Min: {np.min(filtered_semantic_scores_adv):.4f}, Max: {np.max(filtered_semantic_scores_adv):.4f}")
    print(f"All Noise Semantic Count (>= 0.7) vs. Adv: {len(filtered_semantic_scores_adv)}")
    print(f"Total Selected Count: {len(selected_dist_values)}")


# File paths
noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/noise/'
prompt_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/'

datasets = ["squad"]
mean = 1

if __name__ == '__main__':
    for dataset_id in datasets:
        input_file = noise_path + f"{dataset_id}.json"
        prompt_file = prompt_path + f"{dataset_id}.json"
        
        with open(input_file, 'r') as file:
            all_noise = json.load(file)
        with open(prompt_file, 'r') as file:
            target = json.load(file)
            
        cnt = 0
        for i in range(0, len(target)):
            item = target[i]
            target_question = item['question']
            adv_question = item['adv']
            goal_sim = cosine_sim_batch(target_question, adv_question)[0]
            # print(f"target_question is: {target_question}")
            
            noise_questions = [noise['question'] for noise in all_noise]
            noise_ids = [noise['id'] for noise in all_noise]

            batch_size = 2048
            all_similarities = []

            for i in tqdm(range(0, len(noise_questions), batch_size)):
                batch_noise_questions = noise_questions[i:i + batch_size]
                batch_similarities = cosine_sim_batch(batch_noise_questions, [adv_question] * len(batch_noise_questions))
                all_similarities.extend(batch_similarities)

            for i, noise in enumerate(all_noise):
                noise["dist"] = all_similarities[i]

            selected_ids = get_normal_distribution(dist_mean=mean, number=500, all_noise=all_noise)
            selected_questions = [all_noise[i]['question'] for i in selected_ids]
            if target_question in selected_questions:
                cnt += 1 
                
            # plot_distribution(all_noise, selected_ids, dataset_id, goal_sim, target_question, adv_question)
            
        print(f"target question selected out: {cnt}")