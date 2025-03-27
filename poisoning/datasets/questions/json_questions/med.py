from datasets import load_dataset
import json


dataset_id = 'keivalya/MedQuad-MedicalQnADataset'
file_name = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/json_questions/' + dataset_id.split('/')[1] + '.json'

if __name__ == '__main__':    
    with open(file_name, "r") as file:
        data = json.load(file)
        
    print(type(data))


