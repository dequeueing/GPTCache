import json
import torch
from tqdm import tqdm
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
tokenizer.pad_token_id = tokenizer.eos_token_id
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
    do_sample=False,
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


datasets = [
    # 'hotpotqa',
    # 'nq',
    # 'trivia',
    'wiki',
]

if __name__ == '__main__':
    for dataset_id in datasets:
                
        # load data
        input_file = f"{dataset_id}.json"
        with open(input_file, 'r') as f:
            data = json.load(f)
                
        total = len(data)
        for item in tqdm(data):
            question = item['question']
            adv = item['black']
            target_answer = item['wrong']
            
            attacker_response = generate(adv)
            injection_success = check_inject_success(attacker_response, target_answer)
            cos_sim = cosine_sim(question, adv)
            sem_score = semantic_score(question, adv)
            euc_dist = euclidean_distance(question, adv)
            
            item['attacker_response'] = attacker_response
            item['injection_success'] = injection_success
            item['cos_sim_black'] = float(cos_sim)
            item['sem_score_black'] = float(sem_score)
            item['euc_dist_black'] = float(euc_dist)


        with open(input_file, 'w') as file:
            json.dump(data, file, indent=4)
            
        # analysis 
        import numpy as np
        sem_scores = [item['sem_score_black'] for item in data]
        cos_sims = [item['cos_sim_black'] for item in data]
        euc_dists = [item['euc_dist_black'] for item in data]
        injection_successes = [item['injection_success'] for item in data]

        # Calculate stats
        def summarize(metric_list):
            return {
                "min": np.min(metric_list),
                "max": np.max(metric_list),
                "avg": np.mean(metric_list)
            }

        print(f"Dataset: {dataset_id}")
        print("Semantic Score:", summarize(sem_scores))
        print("Cosine Similarity:", summarize(cos_sims))
        print("Euclidean Distance:", summarize(euc_dists))
        print(f"Injection Success Rate: {np.mean(injection_successes):.4f} ({sum(injection_successes)}/{len(injection_successes)})")
