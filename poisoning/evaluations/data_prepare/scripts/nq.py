from datasets import load_dataset

ds = load_dataset("google-research-datasets/natural_questions", "dev")

print(ds)
print(ds['validation'])
ds = ds['validation']

all_question = [item['question']['text'] for item in ds]
print(len(all_question))


import random
some = random.sample(all_question, 10)
for item in some:
    print(item)
    
output_path = "nq.json"
import json
with open(output_path, 'w') as file:
    json.dump(all_question, file, indent=4)