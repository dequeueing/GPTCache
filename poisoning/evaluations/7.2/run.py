import json
import torch
import shutil
import os
from util import *
from typing import *
from util import *

def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = cached_llm.invoke(prompt, cache_obj=the_cache)
    return completion

def get_config(config:str, value):
    (threshold, top_k, noise_number) = (default['thresholds'], default['top_k'], default['noise_number'])
    if config == 'thresholds':
        threshold = value
    if config == 'top_k':
        top_k = value
    if config == 'noise_number':
        noise_number = value
    return threshold, top_k, noise_number

def setup_cache(threshold, top_k, noise_number, dataset_id):
    global cached_llm
    llm = HuggingFacePipeline(pipeline=pipe)
    cached_llm = LangChainLLMs(llm=llm)

    # Clean up the index
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    shutil.rmtree(data_dir)

    # Init cache
    embedding=Huggingface()
    data_manager = manager_factory(
                "sqlite,faiss",
                data_dir=data_dir,
                vector_params={"dimension": embedding.dimension, "top_k": top_k},
                eviction_params={'max_size': max(noise_number, 1000)}, 
            )
    init_similar_cache(
        data_dir=data_dir,
        cache_obj=the_cache,
        pre_func=extract_user_question,
        embedding=embedding,
        data_manager=data_manager,
        evaluation=SbertCrossencoderEvaluation(),
        post_func=from_list,
        config=Config(similarity_threshold=threshold),
    )
    
    # inject noise
    noise_file = f"{noise_path}{dataset_id}_{noise_number}noise.json"
    with open(noise_file, 'r') as f:
        noise_questions = json.load(f)
        
    for question in noise_questions:
        put(question, f"This is a noise question: {question}", cache_obj=the_cache)

    # Check the questions are in the cache
    vector_db = the_cache.data_manager.v  # Access the FAISS vector store
    vector_count = vector_db.count()
    assert vector_count == len(noise_questions)

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/results/'
noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/noise/'
datasets = {
    # "squad": "squad_targeted.json",
    # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = [
    # 'dont_answer_PI_', 
    'ignore_PI_'
]

configs = {
    'thresholds': [0.2, 0.4, 0.6, 0.8, 0.9],  
    'top_k': [1, 3, 5, 10],
    'noise_number': [0, 500, 1000, 2000],
}

default = {
    'thresholds': 0.8,
    'top_k': 5,
    'noise_number': 500,
}


if __name__ == '__main__':
    stat = []
    stat_file = output_path + f"E72_summary.json"
    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            for config in configs:
                for value in configs[config]:
                    independent_var = value
                    threshold, top_k, noise_number = get_config(config, independent_var)
                    setup_cache(threshold, top_k, noise_number, dataset_id)                    
                    
                    input_file = input_path + f"{pattern}{dataset_id}.json"
                    output_file = output_path + f"E72_{pattern}{dataset_id}_{config}{independent_var}.json"

                    with open(input_file, 'r') as f:
                        data = json.load(f)
                                       
                    generated = set()
                    
                    attack_cnt = 0
                    injection_cnt = 0
                    similar_cnt = 0
                    total = len(data)

                    for item in data:
                        if 'attack success' in item:
                            continue
                        
                        injection_success = False
                        attack_success = False
                        similar_enough = False
                        
                        question = item['question']
                        adv = item['adv']
                        target_answer = item['wrong']
                        
                        attacker_response = generate(adv)
                        victim_response = generate(question)
                        if attacker_response in generated or victim_response in generated:
                            setup_cache(threshold, top_k, noise_number, dataset_id)
                            attacker_response = generate(adv)
                            victim_response = generate(question)

                    
                        # if attacker_response == target_answer:
                        if target_answer in attacker_response:
                            injection_success = True
                                            
                        if victim_response == attacker_response:
                            similar_enough = True
                        
                        if similar_enough and injection_success:
                            attack_success = True
                            
                        if attack_success:
                            attack_cnt += 1
                        if injection_success:
                            injection_cnt += 1
                        if similar_enough:
                            similar_cnt += 1
                                                    
                        generated.add(attacker_response)
                        generated.add(victim_response)
                        
                        item['attacker response'] = attacker_response
                        item['victim response'] = victim_response
                        item['attack success'] = attack_success 
                        item['injection success'] = injection_success
                        item['similar enough'] = similar_enough
                            
                        # store adv to local
                        with open(output_file, "w") as file:
                            json.dump(data, file, indent=4)
                            
                            
                    stat.append(
                        {
                        'Dataset': f"E72_{pattern}{dataset_id}_{config}{independent_var}", 
                        'ASR': attack_cnt / total,
                        'total': total,
                        'attack success': attack_cnt,
                        'injection success': injection_cnt,
                        'similar success': similar_cnt,
                        }
                    )
                            
                    with open(stat_file, 'a') as file:
                        json.dump(stat, file, indent=4)
