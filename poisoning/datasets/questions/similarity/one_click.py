import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Union, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from dataclasses import dataclass



datasets = [
    'microsoft/ms_marco',
    'rajpurkar/squad',
    'keivalya/MedQuad-MedicalQnADataset'
]
THRESHOLD = 0.8



if __name__ == '__main__':    
    
    for dataset_id in datasets:
    
        file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/embedding/' + dataset_id.split('/')[1] + '.json'
        new_file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/similarity/' + dataset_id.split('/')[1] + '.json'
        stat_file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/similarity/' + dataset_id.split('/')[1] + '_stat.json'

        with open(file_name, "r") as file:
            data = json.load(file)
        
        embeddings = torch.tensor([item['embedding'] for item in data], dtype=torch.float32).cuda()
        questions = [item['question'] for item in data]
        

        # Vectorized computations on GPU    
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)     # Normalize embeddings for cosine similarity
        cos_sim_matrix = embeddings_norm @ embeddings_norm.T  # GPU matrix multiplication
        euclidean_dist_matrix = torch.cdist(embeddings, embeddings).cuda()  # GPU Euclidean distance

        result = []
        stat = []
        for i in tqdm(range(len(data)), desc="Processing Questions"):
            entry = {'question': questions[i], 'candidates': []}
            mask = (cos_sim_matrix[i] > THRESHOLD) & (torch.arange(len(data), device='cuda') != i)
            indices = mask.nonzero(as_tuple=False).squeeze()

            if indices.numel() > 0:  # If there are candidates
                # Extract relevant pairs
                cos_sims = cos_sim_matrix[i, indices].cpu().numpy()
                euclidean_dists = euclidean_dist_matrix[i, indices].cpu().numpy()
                candidate_questions = [questions[j.item()] for j in indices]

                for j, cos_sim, euc_dist, q in zip(
                    indices, cos_sims, euclidean_dists, candidate_questions
                ):
                    candidate = {
                        'question': q,
                        'cosine_similarity': float(cos_sim),
                        'euclidean_distance': float(euc_dist)
                    }
                    entry['candidates'].append(candidate)

            result.append(entry)
            stat.append({'question': questions[i], 'similar queries #': len(entry['candidates'])})

        with open(new_file_name, "w") as file:
            json.dump(result, file)
            
        with open(stat_file_name, "w") as file:
            json.dump(stat, file)

        
        