import json

def print_item(item):
    for x in item:
        print(f"{x}: {item[x]}")
    print()


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/results_black/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/failure_analysis/'
datasets = {
    # "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    # "ms_marco": "ms_marco_targeted.json"
}

if __name__ == '__main__':
    for dataset_id in datasets:
        input_file = input_path + f"poisoned_{dataset_id}.json"
        output_file = output_path + f"failed_{dataset_id}.json"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
                            
        # fail due to unsimilar q target and q adv
        fail = [item for item in data if not item['attack success'] and item['injection success']]                
                
        with open(output_file, 'w') as file:
            json.dump(fail, file, indent=4)
        
