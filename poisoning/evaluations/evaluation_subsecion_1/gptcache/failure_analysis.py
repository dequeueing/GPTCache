import json

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/results/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/retest/'
datasets = {
    "squad": "squad_targeted.json",
    # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    # "ms_marco": "ms_marco_targeted.json"
}


def print_item(item):
    for x in item:
        print(f"{x}: {item[x]}")
    print()


if __name__ == '__main__':
    for dataset_id in datasets:
        input_file = input_path + f"poisoned_{dataset_id}.json"
        output_file = output_path + f"retest_{dataset_id}.json"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        attacker_responses = set()
        for item in data:
            attacker_responses.add(item['attacker response'])
        
        retest = []
        for item in data:
            if not item['injection success'] and item['attacker response'] in attacker_responses:
                retest.append(item)
                print_item(item)
                
                
        print(f"failure number due to noise inference: {len(retest)}")
        with open(output_file, 'w') as file:
            json.dump(retest, file, indent=4)
        
