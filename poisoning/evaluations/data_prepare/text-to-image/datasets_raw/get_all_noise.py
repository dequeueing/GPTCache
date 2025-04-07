import json
import random

def get_all_noise(input_path, output_path):
    with open(input_path) as file:
        data = json.load(file)
        
    all_noise = [item['captions'][0] for item in data]
    
    with open(output_path, 'w') as f:
        json.dump(all_noise, f, indent=2)
        
# TODO: exclude the noise in the target
def sample_noise(input_path, output_path, noise_number):
    target_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/target_with_white_fix.json"
    with open(target_file) as file:
        target = json.load(file)
    target = [item['prompt'] for item in target]

    # get all noise
    with open(input_path) as file:
        data = json.load(file)
    
    # sample some
    data = random.sample(data, noise_number + 500)
    
    # exclude the target
    data = [item for item in data if item not in target]
    data = random.sample(data, noise_number)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
def check():
    noise_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/noise.json"
    target_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/target_with_white_fix.json"
    with open(target_file) as file:
        target = json.load(file)
    target = [item['prompt'] for item in target]
    
    # get noise 
    with open(noise_path) as file:
        noise = json.load(file)
        
    # check duplicate 
    for target_question in target:
        for noise_question in noise:
            assert target_question != noise_question
            
    print("check completed")
    

    




output_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/noise.json"
input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/all_noise.json"

if __name__ == '__main__':
    # sample_noise(input_path, output_path, noise_number=500)
    check()