from datasets import load_dataset

ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext")
print(len(ds))
print(ds)


# Check the structure of the dataset
print(ds)
ds = ds['validation']


print(len(ds))
# print(ds[0])

questions = [(item['question'], item['answer']['aliases'][0]) for item in ds]
# answers = [item['answer']['aliases'] for item in 

all_questions = []
for item in ds:
    entry = {
        'question': item['question'],
        'correct': item['answer']['aliases'][0]
    }
    all_questions.append(entry)
    
output_path = "trivia.json"
import json
with open(output_path, 'w') as file:
    json.dump(all_questions, file, indent=4)


import random
some = random.sample(questions, 10)
for item in some:
    print(item)