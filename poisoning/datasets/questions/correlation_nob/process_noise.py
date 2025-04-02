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

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/noise/'
prompt_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/correlation_nob/noise/'
datasets = [
    "squad",
    "MedQuad-MedicalQnADataset",
    "ms_marco",
]

if __name__ == '__main__':
    for dataset_id in datasets:
        input_file = input_path + f"{dataset_id}.json"
        prompt_file = prompt_path + f"{dataset_id}.json"
        output_file = output_path + f"{dataset_id}.json"
        
        # load noise and target
        with open(input_file, 'r') as file:
            all_noise = json.load(file)
            
        new = []
        for i, noise in enumerate(all_noise):
            entry = {}
            entry['id'] = i
            entry['question'] = noise
            new.append(entry)
        
        
        with open(output_file, 'w') as file:
            json.dump(new, file, indent=4)
