import numpy as np
import torch
import json
from tqdm import tqdm
from typing import Union, Dict, Tuple, List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

THRESHOLD = 0.8
datasets = [
    # 'microsoft/ms_marco',
    # 'rajpurkar/squad',
    'keivalya/MedQuad-MedicalQnADataset'
]

class SemanticEvaluator:
    def __init__(self, batch_size: int = 8192, device: str = None):
        model_id = "cross-encoder/quora-distilroberta-base"
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() and device is None else device
        print(f"Using device: {self.device}")
        self.encoder = CrossEncoder(model_id, device=self.device)
        self.batch_size = batch_size
        
    def predict(self, q1: Union[str, List[str]], q2: Union[str, List[str]]) -> Union[float, np.ndarray]:
        if isinstance(q1, str) and isinstance(q2, str):
            score = self.encoder.predict([(q1, q2)])[0]
            return float(score)
        elif isinstance(q1, list) and isinstance(q2, list):
            if len(q1) != len(q2):
                raise ValueError("q1 and q2 lists must have the same length")
            sentence_pairs = list(zip(q1, q2))
            scores = []
            for i in range(0, len(sentence_pairs), self.batch_size):
                batch = sentence_pairs[i:i + self.batch_size]
                batch_scores = self.encoder.predict(batch, convert_to_numpy=True)
                scores.extend(batch_scores)
            return np.array(scores)
        else:
            raise TypeError("q1 and q2 must both be strings or both be lists")

if __name__ == "__main__":
    evaluator = SemanticEvaluator(batch_size=512)

    for dataset_id in tqdm(datasets, desc="Processing datasets"):
        input_file = f'/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/embedding/{dataset_id.split("/")[1]}.json'
        output_file = f'/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/distribution/semantic_score/{dataset_id.split("/")[1]}_semantic_scores.json'

        # Load data
        with open(input_file, "r") as file:
            data = json.load(file)

        # Get embeddings and questions
        embeddings = torch.tensor([item['embedding'] for item in data], dtype=torch.float32).cuda()
        questions = np.array([item['question'] for item in data])

        # Normalize embeddings and compute cosine similarity matrix
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        cos_sim_matrix = embeddings_norm @ embeddings_norm.T

        # Randomly select 10% of questions
        num_questions = len(data)
        sample_size = max(1, int(num_questions * 0.1))
        sampled_indices = np.random.choice(num_questions, size=sample_size, replace=False)

        # Prepare output data
        stat = []
        with torch.no_grad():
            self_mask = torch.eye(num_questions, device='cuda', dtype=torch.bool)
            for idx in tqdm(sampled_indices, desc=f"Evaluating {dataset_id}"):
                # Find similar questions based on cosine similarity
                similar_mask = (cos_sim_matrix[idx] > THRESHOLD) & ~self_mask[idx]
                similar_indices = torch.where(similar_mask)[0].cpu().numpy()
                similar_questions = questions[similar_indices].tolist()

                if similar_questions:
                    # Prepare batch for semantic evaluation
                    q1_list = [questions[idx]] * len(similar_questions)
                    q2_list = similar_questions

                    # Get semantic scores using batch inference
                    semantic_scores = evaluator.predict(q1_list, q2_list)

                    # Count questions exceeding semantic threshold
                    semantic_similar_count = np.sum(semantic_scores > THRESHOLD)

                    # Store results
                    stat.append({
                        'question': questions[idx],
                        'cosine embedding similar': len(similar_questions),
                        'semantic score similar': int(semantic_similar_count)
                    })
                else:
                    stat.append({
                        'question': questions[idx],
                        'cosine embedding similar': 0,
                        'semantic score similar': 0
                    })
    
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(stat, f, indent=4)
        print(f"Saved semantic scores to {output_file}")