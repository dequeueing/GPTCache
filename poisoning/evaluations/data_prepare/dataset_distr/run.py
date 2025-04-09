import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
embedding_model_type = 'distilbert-base-uncased'
embedding_model = AutoModel.from_pretrained(embedding_model_type).to('cuda').eval()
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)

# Sentence embedding function
def _sent_embed_from_hidden(hidden_states, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float().to(hidden_states.device)
    sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sentence_embs

# Precompute embeddings
def get_all_embeddings(questions, batch_size=2048):
    embeddings = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Computing embeddings"):
        batch = questions[i:i + batch_size]
        tokenized = embedding_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            hidden_states = embedding_model(**tokenized).last_hidden_state
            batch_embs = _sent_embed_from_hidden(hidden_states, tokenized['attention_mask'])
        embeddings.append(batch_embs.cpu())
    return torch.cat(embeddings, dim=0)

# Compute all similarities
def compute_all_similarities(embeddings):
    embeddings = embeddings.to('cuda')
    norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
    triu_indices = torch.triu_indices(len(embeddings), len(embeddings), offset=1, device='cuda')
    similarities = sim_matrix[triu_indices[0], triu_indices[1]].cpu().numpy()
    return similarities

# Optimized plotting function with dataset name
def plot_similarity_distribution(similarities, output_path, dataset_id):
    sample_size = min(1000000, len(similarities))
    sampled_sim = np.random.choice(similarities, sample_size, replace=False)

    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    hist, bins = np.histogram(sampled_sim, bins=50, density=True)
    plt.bar(bins[:-1], hist, width=np.diff(bins), color='blue', alpha=0.7)
    plt.title(f'Similarity Distribution - {dataset_id}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')

    # CDF
    plt.subplot(1, 2, 2)
    sorted_sim = np.sort(sampled_sim)
    cdf = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
    plt.plot(sorted_sim, cdf, color='green')
    plt.title(f'Cumulative Distribution Function - {dataset_id}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.savefig(f'{output_path}{dataset_id}_distribution.png')
    plt.close()
    
def sample(similarities):
    sample_size = min(1000000, len(similarities))
    sampled_sim = np.random.choice(similarities, sample_size, replace=False).tolist()
    return sampled_sim


# datasets = {
#     # "squad": "squad_targeted.json",
#     # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
#     # "ms_marco": "ms_marco_targeted.json",
#     "text-image": "ms_marco_targeted.json",
# }

datasets = [
    'hotpotqa',
    'nq',
    'trivia',
    'wiki',
]

def sample_target():
    input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/all_noise/'
    output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/plots/'
    data_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/target/'
    
    for dataset_id in datasets:
        input_file = input_path + f"{dataset_id}.json"
        target_file = data_path + f"{dataset_id}.json"

        with open(input_file, 'r') as file:
            all_noise = json.load(file)
            
        import random
        target = random.sample(all_noise, 100)
        
        with open(target_file, 'w') as f:
            json.dump(target, f, indent=2)


        


# Main execution
if __name__ == '__main__':
    # sample_target()
    input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/all_noise/'
    output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/plots/'
    data_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/'
    
    data = []
    for dataset_id in datasets:
        input_file = input_path + f"{dataset_id}.json"
        data_file = data_path + f"{dataset_id}_data.json"

        with open(input_file, 'r') as file:
            all_noise = json.load(file)

        try:
            questions = [item['question'] for item in all_noise]
        except Exception:
            questions = all_noise
            
        print(f"Total questions: {len(questions)}")

        # Precompute embeddings
        embeddings = get_all_embeddings(questions, batch_size=2048)
        similarities = compute_all_similarities(embeddings)
        
        plot_similarity_distribution(similarities, output_path, dataset_id)
        print(f"Plots saved to {output_path}{dataset_id}_distribution.png")
        
        # samples = sample(similarities)        
        # with open(data_file, 'w') as f:
        #     json.dump(samples, f, indent=2)


