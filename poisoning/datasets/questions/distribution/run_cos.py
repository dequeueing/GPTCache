"""Get the number of high cosine similarity candidate for each question"""
from tqdm import tqdm
import numpy as np
import torch
import json


THRESHOLD = 0.8
datasets = [
    'microsoft/ms_marco',
    'rajpurkar/squad',
    'keivalya/MedQuad-MedicalQnADataset'
]
            
if __name__ == '__main__':
    for dataset_id in tqdm(datasets):
        name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/embedding/' + dataset_id.split('/')[1] + '.json'
        file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/embedding/' + dataset_id.split('/')[1] + '.json'
        write_file = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/distribution/' + dataset_id.split('/')[1] + '_cos_sim.json'
        
        # load data
        with open(file_name, "r") as file:
            data = json.load(file)

        # get embeddings and questions
        embeddings = torch.tensor([item['embedding'] for item in data], dtype=torch.float32).cuda()
        questions = np.array([item['question'] for item in data])

        # Vectorized computations on GPU
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)  # Skip if pre-normalized
        cos_sim_matrix = embeddings_norm @ embeddings_norm.T

        # Count similar questions per row
        with torch.no_grad():
            self_mask = torch.eye(len(data), device='cuda', dtype=torch.bool)
            full_mask = (cos_sim_matrix > THRESHOLD) & ~self_mask
            similar_counts = full_mask.sum(dim=1).cpu().numpy()  # Count True values per row

        # Build stat directly
        stat = [
            {'question': questions[i], 'similar queries #': int(similar_counts[i])}
            for i in range(len(data))
        ]

        # Optional: Write to file
        with open(write_file, "w") as file:
            json.dump(stat, file)