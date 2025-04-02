import json
import os
import shutil
import threading
from threading import Lock

from util_para import *
from typing import *
from gptcache import Cache
from gptcache.manager import manager_factory
from gptcache.embedding import Huggingface
from gptcache.adapter.api import init_similar_cache, put
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from gptcache.adapter.langchain_models import LangChainLLMs
from langchain_huggingface import HuggingFacePipeline
from gptcache.config import Config

# Global model setup
device = 'cuda'
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(device)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=pipe)
cached_llm = LangChainLLMs(llm=llm)

# Thread-safe lock for cached_llm access
lock = Lock()

prompt_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/prompts/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/result_new/'
noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/noise_id/'
datasets = {
    "squad": "squad_targeted.json",
}
prompt_injection_patterns = [
    'ignore_no_repeat',
]
configs = {
    'correlation': [0.9, 1.0],
}
default = {
    'thresholds': 0.8,
    'top_k': 5,
    'noise_number': 500,
    'correlation': 0.7
}

def inject_noise(target_question: str, correlation, noise_number, all_noise, the_cache):
    noise_questions = [noise['question'] for noise in all_noise]
    batch_size = 2048
    all_similarities = []
    for i in range(0, len(noise_questions), batch_size):
        batch_noise_questions = noise_questions[i:i + batch_size]
        batch_similarities = cosine_sim_batch(batch_noise_questions, [target_question] * len(batch_noise_questions))
        all_similarities.extend(batch_similarities)
    for i, noise in enumerate(all_noise):
        noise["dist"] = all_similarities[i]
    selected_ids = get_normal_distribution(dist_mean=correlation, number=noise_number, all_noise=all_noise)
    selected_noise = [noise_questions[id] for id in selected_ids]
    for target_question in selected_noise:
        put(target_question, f"This is a noise question: {target_question}", cache_obj=the_cache)
    vector_db = the_cache.data_manager.v
    assert vector_db.count() == len(selected_noise)

def setup_cache(threshold, top_k, noise_number, task_id):
    data_dir = f'.attack_{task_id}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        shutil.rmtree(data_dir)
        os.makedirs(data_dir)
    the_cache = Cache()
    embedding = Huggingface()
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
    return the_cache

def generate(prompt, cache):
    prompt = after_template(prompt, tokenizer)
    with lock:  # Ensure thread-safe access to cached_llm
        completion = cached_llm.invoke(prompt, cache_obj=cache)
    return completion.rstrip('.')

def get_config(config, independent_var):
    threshold = default['thresholds']
    top_k = default['top_k']
    noise_number = default['noise_number']
    correlation = default['correlation']
    if config == 'correlation':
        correlation = independent_var
    return threshold, top_k, noise_number, correlation

def process_config(args):
    pattern, dataset_id, config, value, task_id = args
    independent_var = value
    threshold, top_k, noise_number, correlation = get_config(config, independent_var)
    
    input_file = prompt_path + f"{pattern}_{dataset_id}.json"
    noise_file = noise_path + f"{dataset_id}.json"
    output_file = output_path + f"E72_{pattern}_{dataset_id}_{config}{independent_var}.json"

    with open(noise_file, 'r') as file:
        all_noise = json.load(file)
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    the_cache = setup_cache(threshold, top_k, noise_number, task_id)
    for item in data:
        if 'attack success' in item:
            continue
        
        question = item['question']
        adv = item['adv']
        target_answer = item['wrong'].rstrip('.')
        
        the_cache = setup_cache(threshold, top_k, noise_number, task_id)
        inject_noise(question, correlation, noise_number, all_noise, the_cache)
        
        attacker_response = generate(adv, the_cache)
        victim_response = generate(question, the_cache)
        
        injection_success = attacker_response == target_answer
        similar_enough = victim_response == attacker_response
        attack_success = similar_enough and injection_success
                
        item['attacker response'] = attacker_response
        item['victim response'] = victim_response
        item['attack success'] = attack_success
        item['injection success'] = injection_success
        item['similar enough'] = similar_enough
    
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)

if __name__ == '__main__':
    tasks = [
        (pattern, dataset_id, config, value, f"{pattern}_{dataset_id}_{config}{value}")
        for pattern in prompt_injection_patterns
        for dataset_id in datasets
        for config in configs
        for value in configs[config]
    ]
    
    # Use threads instead of processes
    threads = []
    for task in tasks:
        t = threading.Thread(target=process_config, args=(task,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()