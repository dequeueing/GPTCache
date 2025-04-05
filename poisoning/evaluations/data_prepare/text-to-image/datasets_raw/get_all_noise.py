import json


input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/flickr30k.json"
output_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/all_noise.json"
if __name__ == '__main__':
    with open(input_path) as file:
        data = json.load(file)
        
    all_noise = [item['captions'][0] for item in data]
    
    with open(output_path, 'w') as f:
        json.dump(all_noise, f, indent=2)
