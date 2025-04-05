import json
import torch
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


def euclidean_distance(query1, query2):
    # Tokenization
    query1_tokenized = embedding_tokenizer(query1, return_tensors="pt", padding=False).to(embedding_model.device)
    query2_tokenized = embedding_tokenizer(query2, return_tensors="pt", padding=False).to(embedding_model.device)
    
    # Forward
    q1_embedding = embedding_model(**query1_tokenized).last_hidden_state
    q2_embedding = embedding_model(**query2_tokenized).last_hidden_state
    
    # Get sentence embedding from layer activation
    q1_sent_emb = _sent_embed_from_hidden(q1_embedding, query1_tokenized['attention_mask']).squeeze(0)
    q2_sent_emb = _sent_embed_from_hidden(q2_embedding, query2_tokenized['attention_mask']).squeeze(0)
    
    # distance = torch.dist(q1_sent_emb, q2_sent_emb, p=2)
    q1_sent_emb_norm = q1_sent_emb / torch.norm(q1_sent_emb, p=2)
    q2_sent_emb_norm = q2_sent_emb / torch.norm(q2_sent_emb, p=2)
    distance = torch.norm(q1_sent_emb_norm - q2_sent_emb_norm, p=2)
    return distance

model_id = "cross-encoder/quora-distilroberta-base"
encoder = CrossEncoder(model_id)

def semantic_score(q1, q2):
    score = encoder.predict([(q1, q2)])[0]
    return score

def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/work_test/filtered.json'

if __name__ == '__main__':
    with open(input_path) as file:
        data = json.load(file)
        
    for item in data:
        before = item['before']
        after = item['after']
        cos_sim = cosine_sim(before, after)
        sem_score = semantic_score(before, after)
        print(f"{cos_sim}, {sem_score}, prompt: {before}")
    