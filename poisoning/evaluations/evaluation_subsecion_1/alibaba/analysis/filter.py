import json
import torch
from typing import *

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/alibaba/analysis/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/alibaba/analysis/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}


if __name__ == '__main__':
    stat = []
    for dataset_id in datasets:
        input_file = output_path + f"prediction_{dataset_id}.json"
        output_file = output_path + f"filter_{dataset_id}.json"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        fail = []
        for item in data:
            cos_sim = item['cos_sim']
            
            if cos_sim < 0.8:
                fail.append(item)
            
            
            with open(output_file, "w") as file:
                json.dump(fail, file, indent=4)



                        