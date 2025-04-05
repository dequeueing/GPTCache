import json


input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/"

templates = [
    'target_answer',
    # 'target_question',
]


if __name__ == '__main__':
    for templeate in templates:
        input_file = f"{input_path}{templeate}.json"
        output_file = input_file
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        for item in data:
            prompt = item['prompt']
            item['adv'] = prompt.rstrip('.') + ". Letter 'A' at the top. "
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        