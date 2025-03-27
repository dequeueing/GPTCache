"""Test the embedding model"""
from typing import Union, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
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

THRESHOLD = 0.8
manager = EmbeddingManager()
dataset_id = 'keivalya/MedQuad-MedicalQnADataset'
file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/embedding/' + dataset_id.split('/')[1] + '.json'
new_file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/similarity/' + dataset_id.split('/')[1] + '.json'
stat_file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/similarity/' + dataset_id.split('/')[1] + '_stat.json'


if __name__ == '__main__':
    with open(file_name, "r") as file:
        data = json.load(file)

    embeddings = torch.tensor([item['embedding'] for item in data], dtype=torch.float32).cuda()
    questions = np.array([item['question'] for item in data])

    # Vectorized computations on GPU
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    cos_sim_matrix = embeddings_norm @ embeddings_norm.T

    # Count similar questions and sample 100 questions
    with torch.no_grad():
        self_mask = torch.eye(len(data), device='cuda', dtype=torch.bool)
        full_mask = (cos_sim_matrix > THRESHOLD) & ~self_mask
        similar_counts = full_mask.sum(dim=1).cpu().numpy()

        num_questions = len(data)
        sampled_indices = np.random.choice(num_questions, size=min(10, num_questions), replace=False)
        sampled_mask = full_mask[sampled_indices, :]
        rows, cols = sampled_mask.nonzero(as_tuple=False).T
        rows = sampled_indices[rows.cpu().numpy()]
        cols = cols.cpu().numpy()

    # Compute semantic scores and count if either score is above threshold
    semantic_above_threshold = np.zeros(num_questions, dtype=int)
    for i, j in tqdm(zip(rows, cols), total=len(rows), desc="Computing Semantic Scores"):
        q1, q2 = questions[i], questions[j]
        score1 = float(manager.semantic_evaluator.predict(q1, q2))
        score2 = float(manager.semantic_evaluator.predict(q2, q1))
        if max(score1, score2) > THRESHOLD:  # Increment if either exceeds threshold
            semantic_above_threshold[i] += 1

    # Build stat only for sampled questions
    stat = [
        {
            'question': questions[i],
            'similar queries #': int(similar_counts[i]),
            'semantic_above_threshold': int(semantic_above_threshold[i])
        }
        for i in sampled_indices
    ]

    # Write to file
    with open('medqa_temp_semantics.json', "w") as file:
        json.dump(stat, file)