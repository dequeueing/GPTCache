"""Analyze the distance btw adv and noise"""
import json
import torch
import numpy as np
import requests
from openai import OpenAI


llm_client = OpenAI(
    api_key="sk-b443a2741b474fa892a84e940abe338a", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
)


def get_embedding(prompt):
    completion = llm_client.embeddings.create(
        model="text-embedding-v2",
        input=prompt,
        encoding_format="float"
    )
    embedding = list(completion.data[0].embedding)
    return embedding


def extract_noise_text(s):
    result = s.split("This is a noise question:")[1].strip()
    return result

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


output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/analysis/'
input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/analysis/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
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


def compute(noise, prompt):
    # Assuming get_embedding returns a list or numpy array, you might need to convert them to tensors
    noise = torch.tensor(get_embedding(noise))
    prompt = torch.tensor(get_embedding(prompt))
    
    cosine_sim = torch.nn.functional.cosine_similarity(noise, prompt, dim=0)
    euclidean_dist = torch.norm(noise - prompt)

    return cosine_sim.item(), euclidean_dist.item()    


if __name__ == '__main__':
    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            for config in configs:
                for value in configs[config]:
                    independent_var = value
                    threshold, top_k, noise_number, correlation = get_config(config, independent_var)
                    
                    input_file = input_path + f"E73_{pattern}_{dataset_id}_{config}{independent_var}.json"
                    output_file =  output_path + f"E73_{pattern}_{dataset_id}_{config}{independent_var}.json"
                    with open(input_file, 'r') as f:
                        data = json.load(f)
                                        # Change: since each target question will have a noise set, 
                    #   we have to inject the noise every time.
                    for item in data:
                        question = item['question']
                        adv = item['adv']
                        item['target_adv'] = compute(adv, question)                        
                                
                        with open(output_file, 'w') as file:
                            json.dump(data, file, indent=4)
