""" Craft black-box adversatial prompts for each question in each dataset"""
import json

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/formatted/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/adv_white/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}

if __name__ == '__main__':
    for dataset_id in datasets:
        formatted_file = input_path + f"poisoned_{dataset_id}.json"
        adv_file = output_path + f"poisoned_{dataset_id}.json"
        
        with open(formatted_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            question = item['question']
            wrong = item['wrong']
            adv = craft_malicious_white_box(question, wrong)
            item['adv'] = adv
                
        # store adv to local
        with open(adv_file, "w") as file:
            json.dump(data, file, indent=4)
            print(f"Poisoned results saved to {adv_file}")
