import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


datasets = [
    # 'click',
    'squad',
    'MedQuad-MedicalQnADataset',
    'ms_marco',
]
input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"
result_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"
output_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"

if __name__ == '__main__':
    all = []
    for dataset_id in datasets:
        # Load your JSON data
        result_file = f"{result_path}{dataset_id}.json"
        with open(result_file) as f:
            data = json.load(f)
            
        all.extend(data)
        
    result_file = f"{result_path}all.json"
    with open(result_file, 'w') as file:
        json.dump(all, file, indent=4)