import json
import numpy as np
import torch
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


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/results_default/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/analysis/'
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


def compute(noise, prompt):
    # Assuming get_embedding returns a list or numpy array, you might need to convert them to tensors
    noise = torch.tensor(get_embedding(noise))
    prompt = torch.tensor(get_embedding(prompt))
    
    cosine_sim = torch.nn.functional.cosine_similarity(noise, prompt, dim=0)
    # euclidean_dist = torch.norm(noise - prompt)

    # return cosine_sim.item(), euclidean_dist.item()    
    return cosine_sim.item()


import json
import os

# Get current directory
current_dir = os.getcwd()
json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]

# Process each JSON file separately
for json_file in json_files:
    if 'failed' not in json_file:
        continue
    file_path = os.path.join(current_dir, json_file)
    failed_attacks = []  # Reset for each file
    
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Handle both single dict and list of dicts
        
    for item in data:     
        # count noise inference
        # if 'This is a noise question' in item.get("attacker response") :
        #     noise_cnt += 1
        
        question = item['question']
        adv = item['adv']
        item['question_adv_cosine_sim'] = compute(question, adv)
            
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

                
    
#     # Save to a new JSON file if there are failed attacks
#     if failed_attacks:
#         output_file = f"failed_{json_file}"
#         with open(output_file, 'w') as f:
#             json.dump(failed_attacks, f, indent=4)
#         print(f"Saved {len(failed_attacks)} items with failed attacks from {json_file} to {output_file}")
#     else:
#         print(f"No failed attacks found in {json_file}")

# print("Processing complete.")

    

# if __name__ == '__main__':
#     for pattern in prompt_injection_patterns:
#         for dataset_id in datasets:
#             for config in configs:
#                 for value in configs[config]:
#                     independent_var = value
#                     threshold, top_k, noise_number, correlation = get_config(config, independent_var)
                    
#                     input_file = input_path + f"E73_{pattern}_{dataset_id}_{config}{independent_var}.json"
#                     output_file =  output_path + f"E73_{pattern}_{dataset_id}_{config}{independent_var}.json"
                            
#                     with open(output_file, 'r') as f:
#                         data = json.load(f)

#                     # Extract numerical pairs
#                     noise_adv = np.array([entry['noise_adv'] for entry in data])
#                     noise_target = np.array([entry['noise_target'] for entry in data])
#                     target_adv = np.array([entry['target_adv'] for entry in data])

#                     # Compute statistics
#                     print(f"{dataset_id}")
#                     for name, arr in [('noise_adv', noise_adv), ('noise_target', noise_target), ('target_adv', target_adv)]:
#                         cos_sim, euclid_dist = arr[:, 0], arr[:, 1]
#                         print(f"{name}:")
#                         print(f"  Cosine Similarity - Mean: {np.mean(cos_sim):.4f}, Std: {np.std(cos_sim):.4f}")
#                         print(f"  Euclidean Distance - Mean: {np.mean(euclid_dist):.4f}, Std: {np.std(euclid_dist):.4f}")
                        

