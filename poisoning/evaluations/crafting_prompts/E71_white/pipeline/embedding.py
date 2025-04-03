import json
from util import *


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/E71_white/jsons_filtered/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/E71_white/jsons_filtered/'
datasets = [
    "squad",
    # "MedQuad-MedicalQnADataset",
    # "ms_marco".
]


if __name__ == "__main__":
    set_seed()
    set_logging()
    
    for dataset_id in datasets:
        input_file = input_path + f"{dataset_id}.json"
        output_file = output_path + f"{dataset_id}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        fail = []
        for item in data:
            if item['sem_score'] < 0.8 or item['cos_sim'] < 0.8:
                fail.append(item)
                
        with open(output_file, "w") as file:
            json.dump(fail, file, indent=4)

