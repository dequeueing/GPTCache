import json
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

def get_config(config:str, value):
    (threshold, top_k, noise_number, correlation) = (default['thresholds'], default['top_k'], default['noise_number'], default['correlation'])
    if config == 'thresholds':
        threshold = value
    if config == 'top_k':
        top_k = value
    if config == 'noise_number':
        noise_number = value
    if config == 'correlation':
        correlation = value
    return threshold, top_k, noise_number, correlation


def get_embedding(prompt, embedding_model):
    completion = llm_client.embeddings.create(
        model=embedding_model,
        input=prompt,
        encoding_format="float"
    )
    embedding = list(completion.data[0].embedding)
    return embedding


def compute(prompt1, prompt2, embedding_model):
    """compute the embedding similarity of two prompts"""
    # Assuming get_embedding returns a list or numpy array, you might need to convert them to tensors
    prompt1 = torch.tensor(get_embedding(prompt1, embedding_model))
    prompt2 = torch.tensor(get_embedding(prompt2, embedding_model))
    
    cosine_sim = torch.nn.functional.cosine_similarity(prompt1, prompt2, dim=0)
    # euclidean_dist = torch.norm(noise - prompt)

    return cosine_sim.item()  


datasets = {
    # "squad": "squad_targeted.json",
    # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = [
    # 'dont_answer_PI_', 
    # 'ignore_PI_',
    'ignore_no_repeat',
]

configs = {
    # 'thresholds': [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],  
    # 'top_k': [1, 3, 5, 10],
    # 'noise_number': [0, 500, 1000, 2000],
    'noise_number': [500],
    # 'correlation': [0.85, 0.9, 0.95,  1.0],
}

default = {
    'thresholds': 0.8,
    'top_k': 5,
    'noise_number': 500,
    'correlation': 0.7
}

embedding_models = [
    'text-embedding-v3',      # 1024
    # 'multimodal-embedding-v1', # 1024
    'text-embedding-v2',       # 1536
    'text-embedding-v1',      # 1536
]


llm_client = OpenAI(
    api_key="sk-b443a2741b474fa892a84e940abe338a", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
)

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/different_embedding/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/different_embedding/'


if __name__ == '__main__':
    for dataset_id in datasets:
        # input_file = f"{input_path}E73_{dataset_id}.json"
        data_file = f"{input_path}{dataset_id}.json"
        input_file = f"{output_path}failed_{dataset_id}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
        with open(data_file, 'r') as file:
            stat = json.load(file)
        
        # v3 = [item['text-embedding-v3'] for item in stat]
        # v2 = [item['text-embedding-v2'] for item in stat]
        # v1 = [item['text-embedding-v1'] for item in stat]
        
        # iterate 
        for index, item in enumerate(data):
            question = item['question']
            adv = item['adv']
            
            if item['attack success'] and item['text-embedding-v1'] < 0.8:
                print(item)
                