from datasets import load_dataset
import json

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/hotpot_raw.json'
with open(input_path, 'r') as file:
    data = json.load(file)

print(len(data))
print(data[0])

all_questions = []
for item in data:
    entry = {
        'question': item['question'],
        'correct': item['answer']
    }
    all_questions.append(entry)

print(all_questions)

import random
some = random.sample(all_questions, 10)
for item in some:
    print(item)
    
    
output_path = "hotpotqa.json"
import json
with open(output_path, 'w') as file:
    json.dump(all_questions, file, indent=4)