import json
import shutil
from typing import *


from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.adapter.api import init_similar_cache, put, get
from gptcache.core import Cache, Config
from gptcache.processor.post import nop
from gptcache.manager import manager_factory
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.similarity_evaluation import *
from gptcache.embedding import (
    Huggingface,
)

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


# Change the craft file each time
noise_file = 'json_files/queries.jsonl'
craft_file = 'json_files/black1741583474.1990209.json'

# json or jsonl
if noise_file.endswith(".jsonl"):
    print("This is a JSONL file.")
elif noise_file.endswith(".json"):
    print("This is a JSON file.")

# Load questions in non-target json
with open(noise_file, 'r') as file:
    noise_data = {i: json.loads(line) for i, line in enumerate(file)}
    
# Load crafted attacker prompt
with open(craft_file, 'r') as file:
    craft_data = json.load(file)
    
# Loop through all tests
target_questions = [item['question'] for item in craft_data.values() ]
noise_questions = [item['text'] for item in noise_data.values() if item['text'] not in target_questions]


# Load the tokenizer and model
device = "cuda:0"
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(device)
print(f"Model loaded into: {device}")

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

llm = HuggingFacePipeline(pipeline=pipe)
cached_llm = LangChainLLMs(llm=llm)


# Clean up the index
import os
if not os.path.exists("./attack"):
    os.makedirs("./attack")
shutil.rmtree("./attack")

# Init cache
data_dir = 'attack'
the_cache = Cache()
embedding=Huggingface()
data_manager = manager_factory(
            "sqlite,faiss",
            data_dir=data_dir,
            vector_params={"dimension": embedding.dimension, "top_k": 5},
            eviction_params={'max_size': 4000}
        )
init_similar_cache(
    data_dir=data_dir,
    cache_obj=the_cache,
    pre_func=extract_user_question,
    embedding=embedding,
    data_manager=data_manager,
    evaluation=SbertCrossencoderEvaluation(),
    # # evaluation=OnnxModelEvaluation(),
    # evaluation=KReciprocalEvaluation(vectordb=Faiss('./none', 3, 10), top_k=2, max_distance = 4.0, positive=False),
    post_func=from_list,
    config=Config(similarity_threshold=0.9),
)

# TODO: exclude queries from chosen target questions


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


def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    
total_attack = len(craft_data)
success_attack = 0
success_prompt_injection = 0
for item in craft_data.values():
    question = item['question']
    correct = item['correct answer']
    wrong = item['incorrect answer']
    attacker = item['attacker_query']
    
    # Inject attacker 
    attacker_template = after_template(attacker, tokenizer)
    answer_attacker = cached_llm.invoke(attacker_template, cache_obj=the_cache)
    
    # victim
    victim = question
    victim_template = after_template(victim, tokenizer)
    answer_victim = cached_llm.invoke(victim_template, cache_obj=the_cache)
    
    print("\n---------------------------------")
    prompt_injection_sucess = (wrong == answer_attacker)
    attack_success = (answer_victim == wrong)
    print(f"The question: {question}, correct: {correct}, target: {wrong}")
    print(f"Prompt injection success: {prompt_injection_sucess}")  # maybe the whole system prompt will be returned
    print(f"Attack success: {attack_success}")
    print(f"Attacker question: {attacker}")
    print(f"Attacker response: {answer_attacker}")
    print(f"Victim answer: {answer_victim}")
    if attack_success:
        success_attack += 1
    if prompt_injection_sucess:
        success_prompt_injection += 1
        
    print("---------------------------------\n")
    
print(f"total attack: {total_attack}, success: {success_attack}, injection: {success_prompt_injection}")
print(f"attack success rate: {success_attack / total_attack}")
print(f"prompt injection success rate: {success_prompt_injection / total_attack}")