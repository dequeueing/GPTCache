import json
import torch
from typing import *
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-b443a2741b474fa892a84e940abe338a", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
)


def get_embedding(prompt):
    completion = client.embeddings.create(
        model="text-embedding-v2",
        input=prompt,
        encoding_format="float"
    )
    embedding = torch.tensor(completion.data[0].embedding)
    return embedding


def cosine_sim(p1, p2):
    embedding1 = get_embedding(p1)
    embedding2 = get_embedding(p2)
    
    return torch.nn.CosineSimilarity(dim=0)(
        embedding1, embedding2
    )
    
def euclidean_distance(p1, p2):
    embedding1 = get_embedding(p1)
    embedding2 = get_embedding(p2)
    return torch.sqrt(torch.sum((embedding1 - embedding2) ** 2))



input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/alibaba/analysis/"
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/alibaba/analysis/'
datasets = {
    # "squad": "squad_targeted.json",
    # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}


if __name__ == '__main__':
    stat = []
    # for dataset_id in datasets:
    #     input_file = input_path + f"ali_short_poisoned_{dataset_id}.json"
    #     output_file = output_path + f"prediction_{dataset_id}.json"
    
    input_file = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/alibaba/analysis/ali_euclidean_squad_fail.json'
    output_file = output_path + f"fail_analysis_squad.json"

    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for item in data:
        question = item['question']
        adv = item['adv']
        
        cos_sim = cosine_sim(question, adv)
        euclidean = euclidean_distance(question, adv)
        item['cos_sim'] = float(cos_sim)
        item['euclidean distance'] = float(euclidean)
        
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)