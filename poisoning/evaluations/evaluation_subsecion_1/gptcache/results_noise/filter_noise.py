import json
import os
import shutil
import random
from typing import *



def load_noise(noise_file: str, target: List[str], noise_num=1000):
    with open(noise_file, 'r') as f:
        data = json.load(f)  

    noise_set = set(data) 
    target_set = set(target)  
    
    noise = list(noise_set - target_set) 

    if len(noise) < noise_num:
        raise ValueError("Not enough noise items available after filtering out the target list!")

    selected_noise = random.sample(noise, noise_num)
    return selected_noise



input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/adv_black/"
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/results_noise/'
noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/noise/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}


if __name__ == '__main__':
    for dataset_id in datasets:
        noise_file = noise_path + f"{dataset_id}_noise.json"
        input_file = input_path + f"short_poisoned_{dataset_id}.json"
        filter_noise_file = noise_path + f"filtered_{dataset_id}_noise.json"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        target = []
        for item in data:
            target.append(item['question'])

        noise_questions = load_noise(noise_file, target)
        for noise in noise_questions:
            duplicate = False
            for target_question in target:
                if noise == target_question:
                    duplicate = True
                    break
                assert not duplicate
                
            # store adv to local
            with open(filter_noise_file, "w") as file:
                json.dump(noise_questions, file, indent=4)
    
              