from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import pipeline
import torch
from typing import *
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
import json
import shutil
import os
import numpy as np
import random



# init
device = "cuda" 
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
    max_new_tokens=5000,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    return_full_text=False,  
)

the_cache = Cache()
data_dir = '.attack'
cached_llm = None


def extract_user_question(data: Dict[str, Any], **_: Dict[str, Any]) -> Any:
    input_string = data.get("prompt")
    
    # Define the markers
    start_marker = "<|end_header_id|>"
    end_marker = "<|eot_id|>"
    
    # Find the positions of the markers
    start_pos = input_string.find(start_marker, input_string.find("user")) + len(start_marker)
    end_pos = input_string.find(end_marker, start_pos)
    
    # Extract the substring between the markers
    if start_pos != -1 and end_pos != -1:
        question = input_string[start_pos:end_pos].strip()
        return question
    else:
        return input_string
    
def from_list(messages: List[Any]) -> Any:
    """No change after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            answer = nop(messages)
            assert answer = messages
    """
    return messages[0]


def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )


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

def inject_noise(all_noise, number):
    all_noise = random.sample(all_noise, number)
    for noise in all_noise:
        put(noise, f"This is a noise question: {noise}", cache_obj=the_cache)


def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = cached_llm.invoke(prompt, cache_obj=the_cache)
    completion = completion.rstrip('.')
    return completion


noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/noise/'
datasets = [
    # 'click',
    'squad',
    # 'MedQuad-MedicalQnADataset',
    # 'ms_marco',
]

cache_patterns = [
    'hit',
    'miss',
]


if __name__ == '__main__':
    for pattern in cache_patterns:
        for id in datasets:
            # load noise
            noise_file = noise_path + f"{id}.json"
            with open(noise_file, 'r') as file:
                all_noise = json.load(file)
            all_noise = [item['question'] for item in all_noise]
            print(len(all_noise))
            
            
            # cache
            threshold = 1.0 if pattern == 'miss' else 0
            setup_cache(threshold, 5, 1000)
            inject_noise(all_noise, 100)
            
            # query 
            import time
            from tqdm import tqdm
            times = []
            query = random.sample(all_noise, 300)
            for question in tqdm(query):
                start = time.time()
                generate(question)
                times.append(time.time() - start)
                
            # Calculate stats
            min_time = min(times)
            max_time = max(times)
            avg_time = sum(times) / len(times)
            
            print(f"\nPattern: {pattern}")
            print(f"Min time: {min_time:.4f}s")
            print(f"Max time: {max_time:.4f}s")
            print(f"Avg time: {avg_time:.4f}s")
            print(f"Total questions: {len(times)}")
                