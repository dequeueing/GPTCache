from datasets import load_dataset
import json

dataset_id = 'keivalya/MedQuad-MedicalQnADataset'
file_name = dataset_id.split('/')[1] + '.json'
ds = load_dataset(dataset_id, split="train")

if __name__ == '__main__':
    questions = []
    print(type(ds))
    first = ds[0]
    # for itme in ds:
    #     print(itme)
    print(first)
    
    
    for item in ds:
        questions.append(item['Question'])
        
    print(len(questions))
    
    with open(file_name, "w") as file:
        json.dump(questions, file)

