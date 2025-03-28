""" Convert into [ {question: '', correct: '', wrong: '' } ]"""


import json


dir_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/wrong_answers/'
write_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/formatted/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}

if __name__ == '__main__':
    for dataset_id in datasets:
        poisoned_file = dir_path + f"poisoned_{dataset_id}.json"
        formatted_file = write_path + f"poisoned_{dataset_id}.json"
        
        with open(poisoned_file, 'r') as f:
            data = json.load(f)
            
        formatted = []
        for item in data.values():
            for inner_item in item:
                entry = {}
                entry['question'] = inner_item[0]
                entry['wrong'] = inner_item[1]
                formatted.append(entry)
                
        # store formatted to local
        with open(formatted_file, "w") as file:
            json.dump(formatted, file, indent=4)
            print(f"Poisoned results saved to {formatted_file}")
