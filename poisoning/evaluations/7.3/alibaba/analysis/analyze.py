import json
import numpy as np



def get_config(config:str, value):
    (threshold, top_k, noise_number, correlation) = (default['thresholds'], default['top_k'], default['noise_number'], default['correlation'])
    if config == 'thresholds':
        threshold = value
    if config == 'top_k':
        top_k = value
    if config == 'noise_number':
        noise_number = value
    if config == 'correlation':
        correlation = value
    return threshold, top_k, noise_number, correlation


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/results_default/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/analysis/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = [
    # 'dont_answer_PI_', 
    # 'ignore_PI_',
    'ignore_no_repeat',
]

configs = {
    # 'thresholds': [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],  
    # 'top_k': [1, 3, 5, 10],
    # 'noise_number': [0, 500, 1000, 2000],
    'noise_number': [500],
    # 'correlation': [0.85, 0.9, 0.95,  1.0],
}

default = {
    'thresholds': 0.8,
    'top_k': 5,
    'noise_number': 500,
    'correlation': 0.7
}



    

if __name__ == '__main__':
    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            for config in configs:
                for value in configs[config]:
                    independent_var = value
                    threshold, top_k, noise_number, correlation = get_config(config, independent_var)
                    
                    input_file = input_path + f"E73_{pattern}_{dataset_id}_{config}{independent_var}.json"
                    output_file =  output_path + f"E73_{pattern}_{dataset_id}_{config}{independent_var}.json"
                            
                    with open(output_file, 'r') as f:
                        data = json.load(f)

                    # Extract numerical pairs
                    noise_adv = np.array([entry['noise_adv'] for entry in data])
                    noise_target = np.array([entry['noise_target'] for entry in data])
                    target_adv = np.array([entry['target_adv'] for entry in data])

                    # Compute statistics
                    print(f"{dataset_id}")
                    for name, arr in [('noise_adv', noise_adv), ('noise_target', noise_target), ('target_adv', target_adv)]:
                        cos_sim, euclid_dist = arr[:, 0], arr[:, 1]
                        print(f"{name}:")
                        print(f"  Cosine Similarity - Mean: {np.mean(cos_sim):.4f}, Std: {np.std(cos_sim):.4f}")
                        print(f"  Euclidean Distance - Mean: {np.mean(euclid_dist):.4f}, Std: {np.std(euclid_dist):.4f}")
                        

