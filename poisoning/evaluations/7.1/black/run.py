import json
import torch
from util import *
from typing import *

from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# Load the tokenizer and model
device = "cuda" 
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(device)

# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    return_full_text=False,  # Only return the generated text, not the input
)


def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = pipe(prompt)
    return completion[0]['generated_text']


def check_inject_success(attacker_response, target_answer):
    attacker_response = attacker_response.rstrip('.')
    target_answer = target_answer.rstrip('.')
    return attacker_response == target_answer


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/results/'
datasets = {
    # "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    # "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = [
    'dont_answer_PI_', 
    # 'ignore_PI_'
]

if __name__ == '__main__':
    stat = []
    stat_file = output_path + f"gptcache_summary.json"
    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            
            input_file = input_path + f"{pattern}{dataset_id}.json"
            output_file = output_path + f"E71_{pattern}{dataset_id}.json"
            
            # load data
            with open(input_file, 'r') as f:
                data = json.load(f)
                 
            total = len(data)
            for item in data:
                injection_success = False
                attack_success = False
                similar_enough = False
                
                question = item['question']
                adv = item['adv']
                target_answer = item['wrong']
                
                attacker_response = generate(adv)
                injection_success = check_inject_success(attacker_response, target_answer)
                cos_sim = cosine_sim(adv, question)
                sem_score = semantic_score(question, adv)
                euc_dist = euclidean_distance(adv, question)
                
                
                item['attacker_response'] = attacker_response
                item['injection_success'] = injection_success
                item['cos_sim'] = float(cos_sim)
                item['sem_score'] = float(sem_score)
                item['euc_dist'] = float(euc_dist)


                with open(output_file, 'w') as file:
                    json.dump(data, file, indent=4)


