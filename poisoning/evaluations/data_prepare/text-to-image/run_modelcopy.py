import time
import json
import random
from diffusers import DiffusionPipeline


model_ids = {
    'stabilityai/stable-diffusion-2-1': 'stable-diffusion-2-1',
    'stabilityai/stable-diffusion-3.5-large': 'stable-diffusion-3.5-large'
}
model_id = 'stabilityai/stable-diffusion-3.5-large'
pipe = DiffusionPipeline.from_pretrained(model_id).to('cuda')


def generate(prompt):
    image = pipe(prompt).images[0]
    return image

files = [
    "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/target_with_white.json",
    # "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/target_questions.json"
]
output_path = f"/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/generation/whitebox-{model_ids[model_id]}/"

if __name__ == '__main__':
    random.seed(42)
    for input_file in files:
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        record = {}
        for index, item in enumerate(data):
            adv = item['white']
            # prompt = item['prompt']
            print(adv)
            image = generate(adv)
            
            # output_file = output_path + model_ids[model_id] + '/' + prompt.rstrip('.')  + ".png"
            output_file = f"{output_path}{index}.png"
            record[index] = adv
            image.save(output_file)
            
        for item in record:
            print(f"{item}: {record[item]}")
        

        