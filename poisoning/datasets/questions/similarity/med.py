import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Union, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from dataclasses import dataclass




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
    questions = [item['question'] for item in data]

    # Vectorized computations on GPU
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)  # Normalize (skip if pre-normalized)
    cos_sim_matrix = embeddings_norm @ embeddings_norm.T  # Cosine similarity
    euclidean_dist_matrix = torch.cdist(embeddings, embeddings)  # Euclidean distance

    # Precompute mask for all rows at once
    self_mask = torch.eye(len(data), device='cuda', dtype=torch.bool)
    full_mask = (cos_sim_matrix > THRESHOLD) & ~self_mask  # Exclude self-comparisons

    result = []
    stat = []
    # Process all rows in one go
    indices_per_row = [row.nonzero(as_tuple=False).squeeze() for row in full_mask]

    for i, indices in tqdm(enumerate(indices_per_row), total=len(data), desc="Processing Questions"):
        entry = {'question': questions[i], 'candidates': []}
        if indices.numel() > 0:
            if indices.dim() == 0:  # Handle single index case
                indices = indices.unsqueeze(0)
            cos_sims = cos_sim_matrix[i, indices].cpu().numpy()
            euclidean_dists = euclidean_dist_matrix[i, indices].cpu().numpy()
            candidate_questions = [questions[j.item()] for j in indices]

            entry['candidates'] = [
                {
                    'question': q,
                    'cosine_similarity': float(cos_sim),
                    'euclidean_distance': float(euc_dist)
                }
                for q, cos_sim, euc_dist in zip(candidate_questions, cos_sims, euclidean_dists)
            ]

        result.append(entry)
        stat.append({'question': questions[i], 'similar queries #': len(entry['candidates'])})

    with open(new_file_name, "w") as file:
        json.dump(result, file)
    with open(stat_file_name, "w") as file:
        json.dump(stat, file)    
    