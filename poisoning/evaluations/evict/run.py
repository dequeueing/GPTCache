# Setting: fifo, squad dataset
# Independent var: the user number, the cache size (prompts required to evict the nosise)
# Dependent var: number of queries needed to send by the attacker

import json
import time
import shutil
import os
import random
import numpy as np
from typing import *
from util import *
from util import the_cache
import matplotlib.pyplot as plt


def get_cache_size():
    global the_cache
    vector_db = the_cache.data_manager.v
    vector_count = vector_db.count()
    return vector_count


def setup_cache(cache_size):
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
                vector_params={"dimension": embedding.dimension, "top_k": 5},
                eviction_params={'max_size': cache_size, 'eviction': 'FIFO'}, 
            )
    init_similar_cache(
        data_dir=data_dir,
        cache_obj=the_cache,
        pre_func=extract_user_question,
        embedding=embedding,
        data_manager=data_manager,
        evaluation=SbertCrossencoderEvaluation(),
        post_func=from_list,
        config=Config(similarity_threshold=0.8),
    )
    
    
def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = cached_llm.invoke(prompt, cache_obj=the_cache)
    return completion


def get_config(config:str, value):
    (user_number, cache_size) = (default['user number'], default['cache size'])
    if config == 'user number':
        user_number = value
    if config == 'cache size':
        cache_size = value
    return user_number, cache_size


# 250 users -> 1 query per second
configs = {
    # 'user number': [250, 500, 1000, 2000],
    'user number': [250],
    # 'cache size': [500, 1000, 1500, 2000]
}
default = {
    'user number': 250,
    'cache size': 500
}



if __name__ == '__main__':
    # set seed
    random.seed(42)
    
    # init cache
    for config in configs:
        for value in configs[config]:
            user_number, cache_size = get_config(config, value)
            
    setup_cache(cache_size)
    
    # load noise
    noise_file = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evict/data/squad.json'
    with open(noise_file, 'r') as f:
        noise = json.load(f)
        
    noise_question = [item['question'] for item in noise]
    noise_id = [item['id'] for item in noise]
    
    # Change: if not hit, will return the same prompt as itself.
        
    # inject noise 
    start = time.time()
    selected_noise_id = set()
    injected_noise_id = set()
    while len(injected_noise_id) < cache_size:
        random_question = random.choice(noise)
        random_question_str = random_question['question']
        random_question_id = random_question['id']
        
        # check 
        if random_question_id in selected_noise_id:
            continue
        selected_noise_id.add(random_question_id)
        
        # try inject
        completion = generate(random_question_str)
        if completion == random_question_str:
            injected_noise_id.add(random_question_id)
            
        print(f"current cache number: {get_cache_size()}")
            
    print(f"injection finished")
    print(f"time taken: {time.time() - start}")
    
    # simulate end users; default: 1 query per second
    # so the user frequency can be summarized as how many queries the user send; and the time varies based on frequency
    # 5000 end user queries
    chosen_end_user_id = set()
    injected_end_user_id = set()  # len of this set is the newly injected into
    attacker_to_inject = []
    for i in range(0, 5000):
        # select user query, excluded from the chosen ones
        while True:
            random_question = random.choice(noise)
            random_question_str = random_question['question']
            random_question_id = random_question['id']
            if random_question_id not in chosen_end_user_id and random_question_id not in selected_noise_id:
                chosen_end_user_id.add(random_question_id)
                break
            
        # inject
        response = generate(random_question_str)
        if response == random_question_str:  # inject success
            injected_end_user_id.add(random_question_id)
        
        # record attacker data 
        attacker_to_inject.append(cache_size - len(injected_end_user_id))
        
        # finish eviction
        if len(injected_end_user_id) == cache_size:
            break
        
    # draw
    plt.plot(attacker_to_inject)
    plt.title("Eviction Evaluation")
    plt.xlabel("Number of user injection")
    plt.ylabel("Number of attacker injection")
    plt.grid(True)
    plt.show()
    plt.savefig("attacker_to_inject_plot.png")
