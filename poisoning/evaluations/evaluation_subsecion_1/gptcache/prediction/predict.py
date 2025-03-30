import json
import torch
from typing import *

from sentence_transformers import CrossEncoder
from transformers import (
    AutoModel,
    AutoTokenizer,
)



embedding_model_type = 'distilbert-base-uncased'
embedding_model = AutoModel.from_pretrained(embedding_model_type).to('cuda')
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)


def _sent_embed_from_hidden(hidden_states, attention_mask):
    """convert token embedding to sentence embedding, mean the dimensions across all tokens"""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )
    input_mask_expanded = input_mask_expanded.to(hidden_states.device)
    sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embs

def cosine_sim(query1, query2):
    # Tokenization
    query1_tokenized = embedding_tokenizer(query1, return_tensors="pt", padding=False).to(embedding_model.device)
    query2_tokenized = embedding_tokenizer(query2, return_tensors="pt", padding=False).to(embedding_model.device)
    
    # Forward
    q1_embedding = embedding_model(**query1_tokenized).last_hidden_state
    q2_embedding = embedding_model(**query2_tokenized).last_hidden_state
    
    # Get sentence embedding from layer activation
    q1_sent_emb = _sent_embed_from_hidden(q1_embedding, query1_tokenized['attention_mask']).squeeze(0)
    q2_sent_emb = _sent_embed_from_hidden(q2_embedding, query2_tokenized['attention_mask']).squeeze(0)
    
    # Cosine similarity
    return torch.nn.CosineSimilarity(dim=0)(
        q1_sent_emb, q2_sent_emb
    )


model_id = "cross-encoder/quora-distilroberta-base"
encoder = CrossEncoder(model_id)

def semantic_score(q1, q2):
    score = encoder.predict([(q1, q2)])[0]
    return score


input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/results_black/"
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/prediction/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}


if __name__ == '__main__':
    stat = []
    for dataset_id in datasets:
        input_file = input_path + f"gptcache_short_poisoned_{dataset_id}.json"
        output_file = output_path + f"prediction_{dataset_id}.json"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            question = item['question']
            adv = item['adv']
            
            cos_sim = cosine_sim(question, adv)
            semantic = semantic_score(question, adv)
            item['cos_sim'] = float(cos_sim)
            item['semantic'] = float(semantic)
            
            with open(output_file, "w") as file:
                json.dump(data, file, indent=4)



                        