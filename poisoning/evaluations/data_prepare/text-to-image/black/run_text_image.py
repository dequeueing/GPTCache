import json
import shutil
import os
import numpy as np
from util import *
from typing import *

def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = cached_llm.invoke(prompt, cache_obj=the_cache)
    completion = completion.rstrip('.')
    return completion

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

def setup_cache(threshold, top_k, noise_number):
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
    
def inject_noise(target_question:str, correlation, noise_number,  all_noise):
    # Extract noise questions and IDs
    # noise_questions = all_noise

    # # Compute cosine similarity in batches
    # batch_size = 2048
    # all_similarities = []

    # for i in range(0, len(noise_questions), batch_size):
    #     batch_noise_questions = noise_questions[i:i + batch_size]
    #     batch_similarities = cosine_sim_batch(batch_noise_questions, [target_question] * len(batch_noise_questions))
        
    #     # Store computed similarities
    #     all_similarities.extend(batch_similarities)

    # # Attach cosine similarity to each noise sample
    # for i, noise in enumerate(all_noise):
    #     noise["dist"] = all_similarities[i]
    # selected_ids = get_normal_distribution(dist_mean=correlation, number=noise_number, all_noise=all_noise)
    # selected_noise = [noise_questions[id] for id in selected_ids]
    
    for target_question in all_noise:
        put(target_question, f"This is a noise question: {target_question}", cache_obj=the_cache)
        
    # Check the questions are in the cache
    vector_db = the_cache.data_manager.v  # Access the FAISS vector store
    vector_count = vector_db.count()
    assert vector_count == len(all_noise)

def get_normal_distribution(dist_mean: float, number: int, all_noise, seed: int = 20):
    """Return a subset of noise IDs where cosine similarity follows a normal distribution."""
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility
    cosine_similarities = np.array([noise['dist'] for noise in all_noise])
    std_dev = np.std(cosine_similarities) if np.std(cosine_similarities) > 0 else 1  # Avoid division by zero
    probabilities = np.exp(-0.5 * ((cosine_similarities - dist_mean) / std_dev) ** 2)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(all_noise), size=number, p=probabilities, replace=False)
    return [all_noise[i]['id'] for i in selected_indices]

datasets = {
    "squad": "squad_targeted.json",
    # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    # "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = [
    # 'dont_answer_PI_', 
    # 'ignore_PI_',
    'ignore_no_repeat',
    # 'only',
]

configs = {
    # 'thresholds': [0.8],  
    # 'thresholds': [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],  
    # 'top_k': [1, 3, 5, 10],
    # 'noise_number': [0, 500, 1000, 2000, 5000],
    # 'correlation': [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    'correlation': [0.85],
}

default = {
    'thresholds': 0.8,
    'top_k': 50,
    'noise_number': 500,
    'correlation': 0.7
}


prompt_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/black/results/'
noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/noise_id/'

if __name__ == '__main__':
    stat_file = output_path + f"text_to_image_summary.json"
    with open(stat_file, 'r') as file:
        stat = json.load(file)

    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            for config in configs:
                for value in configs[config]:
                    independent_var = value
                    threshold, top_k, noise_number, correlation = get_config(config, independent_var)
                    
                    # prompt_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/target_answer.json"
                    prompt_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets/target_with_white.json"
                    noise_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/text-to-image/datasets_raw/noise.json"
                    output_file = output_path + f"E73_text_image_white_{pattern}_{dataset_id}_{config}{independent_var}.json"

                    # load noise and target
                    with open(noise_file, 'r') as file:
                        all_noise = json.load(file)
                    with open(prompt_file, 'r') as f:
                        data = json.load(f)
                                       
                    
                    attack_cnt = 0
                    injection_cnt = 0
                    similar_cnt = 0
                    total = len(data)

                    # Change: since each target question will have a noise set, 
                    #   we have to inject the noise every time.
                    
                    # data = [
                    #     {
                    #         "prompt": " A man with a black t-shirt is holding a yellow electric guitar on stage at a music event .",
                    #         "adv": " A man with a black t-shirt is holding a yellow electric guitar on stage at a music event . Letter 'A' at the top. ",
                    #         "white": " A man with a black t-shirt is holding a yellow electric guitar on stage at a music event . Letter 'A' at the top. \u2026\u2026\u2026\u2026ub",
                    #     }
                    # ]
                    
                    for item in data:
                        if 'attack success' in item:
                            continue
                        
                        
                        injection_success = False
                        attack_success = False
                        similar_enough = False
                        
                        question = item['prompt']
                        adv = item['white']
                        

                        setup_cache(threshold, top_k, noise_number)
                        inject_noise(question, correlation, noise_number,  all_noise)
                        
                        attacker_response = generate(adv)
                        victim_response = generate(question)
                        # if attacker_response in generated or victim_response in generated:
                        #     setup_cache(threshold, top_k, noise_number, dataset_id)
                        #     attacker_response = generate(adv)
                        #     victim_response = generate(question)

                    
                        # if attacker_response == target_answer:
                        if "This is a noise question:" not in attacker_response:
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
                        'attack success': attack_cnt / total,
                        'injection success': injection_cnt / total,
                        'similar success': similar_cnt / total,
                        }
                    )
                            
                    with open(stat_file, 'w') as file:
                        json.dump(stat, file, indent=4)
