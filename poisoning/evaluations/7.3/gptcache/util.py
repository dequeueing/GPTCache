from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import pipeline
import torch
from typing import *
import sys
sys.path.insert(0, '/home/taojie_wang@idm.teecertlabs.com/GPTCache')
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.adapter.api import init_similar_cache, put
from gptcache.core import Cache, Config
from gptcache.processor.post import nop
from gptcache.manager import manager_factory
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.similarity_evaluation import *
from gptcache.embedding import (
    Huggingface,
)


# Load model and tokenizer
embedding_model_type = 'distilbert-base-uncased'
embedding_model = AutoModel.from_pretrained(embedding_model_type).to('cuda').eval()
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)

# Function for sentence embedding
def _sent_embed_from_hidden(hidden_states, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    ).to(hidden_states.device)
    sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embs
    
# Optimized cosine similarity function
def cosine_sim_batch(queries1, queries2):
    queries1_tokenized = embedding_tokenizer(queries1, return_tensors="pt", padding=True, truncation=True).to('cuda')
    queries2_tokenized = embedding_tokenizer(queries2, return_tensors="pt", padding=True, truncation=True).to('cuda')
    
    with torch.no_grad():
        q1_embedding = embedding_model(**queries1_tokenized).last_hidden_state
        q2_embedding = embedding_model(**queries2_tokenized).last_hidden_state

        q1_sent_emb = _sent_embed_from_hidden(q1_embedding, queries1_tokenized['attention_mask'])
        q2_sent_emb = _sent_embed_from_hidden(q2_embedding, queries2_tokenized['attention_mask'])
        
        return torch.nn.functional.cosine_similarity(q1_sent_emb, q2_sent_emb).cpu().numpy()



# Load the tokenizer and model
device = "cuda" 
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(device)

# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    do_sample=False,
    return_full_text=False,  
)


def extract_user_question(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    input_string = data.get("prompt")
    
    # Define the markers
    start_marker = "<|end_header_id|>"
    end_marker = "<|eot_id|>"
    
    # Find the positions of the markers
    start_pos = input_string.find(start_marker, input_string.find("user")) + len(start_marker)
    end_pos = input_string.find(end_marker, start_pos)
    
    # Extract the substring between the markers
    if start_pos != -1 and end_pos != -1:
        question = input_string[start_pos:end_pos].strip()
        return question
    else:
        return input_string
    
def from_list(messages: List[Any]) -> Any:
    """No change after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            answer = nop(messages)
            assert answer = messages
    """
    return messages[0]


def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )



the_cache = Cache()
data_dir = '.attack'
cached_llm = None




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



if __name__ == '__main__':
    p1 = 'what is your name?'
    p2 = 'Tell me your name?'
    print(semantic_score(p1, p2))
    print(euclidean_distance(p1, p2))
    print(cosine_sim(p1, p2))
    
