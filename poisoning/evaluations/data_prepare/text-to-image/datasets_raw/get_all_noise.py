import json
import random

def get_all_noise(input_path, output_path):
    with open(input_path) as file:
        data = json.load(file)
        
    all_noise = [item['captions'][0] for item in data]
    
    with open(output_path, 'w') as f:
        json.dump(all_noise, f, indent=2)
        
def sample_noise(input_path, output_path, noise_number):
    with open(input_path) as file:
        data = json.load(file)
        
    data = random.sample(data, noise_number)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)



output_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/noise.json"
input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/all_noise.json"

if __name__ == '__main__':
    sample_noise(input_path, output_path, noise_number=500)