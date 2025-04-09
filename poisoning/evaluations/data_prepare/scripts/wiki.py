from datasets import load_dataset

ds = load_dataset("microsoft/wiki_qa")
print(ds)

ds = ds['train']

print(ds[0])

all_questions = []
for item in ds:
    if not item['label']:
        continue
    entry = {
        'question': item['question'],
        'correct': item['answer']
    }
    all_questions.append(entry)


print(f"len: {len(all_questions)}")
import random
some = random.sample(all_questions, 10)
for item in some:
    print(item)
    
    
output_path = "wiki.json"
import json
with open(output_path, 'w') as file:
    json.dump(all_questions, file, indent=4)