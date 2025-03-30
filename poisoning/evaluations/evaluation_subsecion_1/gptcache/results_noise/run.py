""" Simulate the situation where there are noise in the cache. """

import json
import os
import shutil
import random
from typing import *
from util import extract_user_question, from_list, after_template

from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

import sys
sys.path.insert(0, '/home/taojie_wang@idm.teecertlabs.com/GPTCache')

from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.adapter.api import init_similar_cache, put
from gptcache.core import Cache, Config
from gptcache.processor.post import nop
from gptcache.manager import manager_factory
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.similarity_evaluation import *
from gptcache.embedding import (
    Huggingface,
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


def init_cache():
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
                eviction_params={'max_size': 20, 'eviction':'FIFO'},   # to avoid interference
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
    
    return cached_llm
    
def inject_noise(noise_questions):
    # Put non-target questions into the cache
    for question in noise_questions:
        put(question, f"This is a noise question: {question}", cache_obj=the_cache)


    # Check the questions are in the cache
    vector_db = the_cache.data_manager.v  # Access the FAISS vector store
    vector_count = vector_db.count()
    index_file_size = os.path.getsize(os.path.join(data_dir, "faiss.index"))
    print(f"Total vectors in FAISS: {vector_count}")
    print(f"FAISS index file size (bytes): {index_file_size}")
    assert vector_count == len(noise_questions)

def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = cached_llm.invoke(prompt, cache_obj=the_cache)
    return completion

def rebuild_cache():
    global cached_llm
    cached_llm = init_cache()

# cache instance
the_cache = Cache()
data_dir = 'attack'
cached_llm = init_cache()


input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/adv_black/"
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/results_black/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}


if __name__ == '__main__':
    stat = []
    stat_file = output_path + f"gptcache_summary.json"
    for dataset_id in datasets:
        rebuild_cache()
        
        input_file = input_path + f"short_poisoned_{dataset_id}.json"
        output_file = output_path + f"gptcache_short_poisoned_{dataset_id}.json"
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        generated = set()
        retest = []
        
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
                rebuild_cache()
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
                
            # concern: whether interference exist?
                
        stat.append(
            {
            'Dataset': dataset_id, 
            'ASR': attack_cnt / total,
            'total': total,
            'attack success': attack_cnt,
            'injection success': injection_cnt,
            'similar success': similar_cnt,
            }
        )
                
        with open(stat_file, 'w') as file:
            json.dump(stat, file, indent=4)
    print(stat)
