# test the effect of image generation with the attacker prompt
import json
from diffusers import DiffusionPipeline

with open('result.json', 'r') as f:
  data = json.load(f)


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda")


def generate(prompt):
    return pipe(prompt=prompt).images[0]

for item in data.values():
    victim = item['victim']
    attacker = item['attacker']
    target = item['target']
    
    image = generate(attacker)
    image.save(f"{target}_{victim}.png")
