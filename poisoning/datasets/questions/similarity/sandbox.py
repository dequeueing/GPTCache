"""Test the embedding model"""
from typing import Union, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
import torch
import json


@dataclass
class EvaluationResult:
    p1 : str 
    p2 : str
    cosine_similarity : float
    semantic_score : Tuple
    euclidean_distance : float
    
class SemanticEvaluator:
    def __init__(self):
        model_id = "cross-encoder/quora-distilroberta-base"
        self.model_id = model_id
        self.encoder = CrossEncoder(model_id)
        
    def predict(self, q1, q2) -> float:
        """Warning: the order of q1 and q2 would affect the result. """
        score = self.encoder.predict([(q1, q2)])[0]
        return score



class EmbeddingManager:
    def __init__(self):
        embedding_model_type = 'distilbert-base-uncased'
        self.embedding_model_type = embedding_model_type
        self.model = AutoModel.from_pretrained(embedding_model_type).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)
        self.semantic_evaluator = SemanticEvaluator()
        
    def _sent_embed_from_hidden(self, hidden_states, attention_mask):
        """convert token embedding to sentence embedding, mean the dimensions across all tokens"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        input_mask_expanded = input_mask_expanded.to(hidden_states.device)
        sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return sentence_embs


    def to_embedding(self, prompt:str, to_list=False) -> Union[list, torch.Tensor]:
        """prompt -> tokenized -> output embedding (sequence length * embedding dim) -> sentence embedding (embedding dim) """
        embedding_tokenizer = self.tokenizer
        embedding_model = self.model
        prompt_tokenized = embedding_tokenizer(prompt, return_tensors="pt", padding=False).to(embedding_model.device)
        output_embedding = embedding_model(**prompt_tokenized).last_hidden_state  # sequence length * embedding dim
        sentence_embedding = self._sent_embed_from_hidden(output_embedding, prompt_tokenized['attention_mask']).squeeze(0)
        
        if to_list:
            sentence_embedding = sentence_embedding.tolist()  
        return sentence_embedding
            
    
    def evaluate_similarity(self, promp1: Dict, promp2: Dict) -> EvaluationResult:
        """evaluate similarity btw two prompts in terms of their embedding

        Args:
            promp1 (Dict): fields: question(str) and embedding(list or torch)
            promp2 (Dict): fields: question(str) and embedding(list or torch)

        Returns:
            EvaluationResult
        """
        # check
        embedding_1 = torch.tensor(promp1['embedding'], dtype=float)
        embedding_2 = torch.tensor(promp2['embedding'], dtype=float)
        question1 = promp1['question']
        question2 = promp2['question']
        
        # cosine similarity 
        cos_sim =  float(torch.nn.CosineSimilarity(dim=0)(embedding_1, embedding_2))

        
        # semantic score
        semantic_score1 = float(self.semantic_evaluator.predict(question1, question2))
        semantic_score2 = float(self.semantic_evaluator.predict(question2, question1))
                        
        # euclidean distance
        euclidean_distance = float(torch.norm(embedding_1 - embedding_2, p=2))
        
        return EvaluationResult(
            question1, question2, cos_sim, (semantic_score1, semantic_score2), euclidean_distance
        )
    
    

manager = EmbeddingManager()

datasets = [
    'microsoft/ms_marco',
    'rajpurkar/squad',
    'keivalya/MedQuad-MedicalQnADataset'
]

dataset_id = 'keivalya/MedQuad-MedicalQnADataset'
# file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/json_questions/' + dataset_id.split('/')[1] + '.json'
file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/embedding/' + dataset_id.split('/')[1] + '.json'


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch

if __name__ == '__main__':
    with open(file_name, "r") as file:
        data = json.load(file)
    
    
    embeddings = torch.tensor([item['embedding'] for item in data], dtype=torch.float32).cuda()
    questions = [item['question'] for item in data]
    THRESHOLD = 0.8
    

    # Vectorized computations on GPU    
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)     # Normalize embeddings for cosine similarity
    cos_sim_matrix = embeddings_norm @ embeddings_norm.T  # GPU matrix multiplication
    euclidean_dist_matrix = torch.cdist(embeddings, embeddings).cuda()  # GPU Euclidean distance

    result = []
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

    