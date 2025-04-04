"""We have to select the best prompt from all our variations. """
import json
import torch
from typing import *

output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/E71_white/results_complete/'
input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/E71_white/results/'
datasets = [
    "squad",
    "MedQuad-MedicalQnADataset",
    "ms_marco",
]

if __name__ == "__main__":    
    for dataset_id in datasets:
        input_file = input_path + f"white_{dataset_id}.json"
        output_file = output_path + f"{dataset_id}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        good = []
        threshold = 0.8
        for item in data:
            entry = {}
            entry['question'] = item['question']
            entry['wrong'] = item['wrong']
            entry['black'] = item['adv_black']
            entry['white'] = item['adv_black']
            entry['black_cos_sim'] = item['cos_sim']
            entry['black_semantic_score'] = item['sem_score']
            entry['white_cos_sim'] = item['cos_sim']
            entry['white_semantic_score'] = item['sem_score']
            
            # white box both better
            if item['cos_sim_final'] >  item['cos_sim'] and  \
                item['sem_score_final'] >  item['sem_score']:
                    entry['white'] = item['adv_final']
            
            # black box fails the threshold
            elif item['sem_score'] < threshold and item['sem_score_final'] > threshold: 
                entry['white'] = item['adv_final']
                
            # both ok. choose the higher one
            elif item['sem_score'] > threshold and item['sem_score_final'] > threshold \
                and item['sem_score_final'] + item['cos_sim_final'] > item['sem_score'] + item['cos_sim']:
                    entry['white'] = item['adv_final']
                    
            if entry['white'] == item['adv_final']:
                entry['white_cos_sim'] = item['cos_sim_final']
                entry['white_semantic_score'] = item['sem_score_final']
  
            good.append(entry)
            
        
        with open(output_file, "w") as file:
            json.dump(good, file, indent=4)
