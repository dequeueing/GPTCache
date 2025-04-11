import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import pipeline
# from testing import *



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
    return_full_text=False,  
)


def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )


def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = pipe(prompt)
    return completion[0]['generated_text']


def generate_similar_question(input_prompt):
    def after_template(message, tokenizer):
        system_prompt = "You are a helpful assistant that generates **one rephrased question** based on the user's input. The new question **must preserve the exact meaning** of the original but use different wording. Do not include any additional text or explanations in your response."
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate one question similar to: {message}"},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    prompt = after_template(input_prompt, tokenizer)
    completion = pipe(prompt)
    return completion[0]['generated_text']




# Load once
model_name = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt_model.eval()

def perplexity(text1: str, text2: str) -> float:
    text = text1 + " " + text2
    inputs = gpt_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()


datasets = [
    # 'click',
    'squad',
    'MedQuad-MedicalQnADataset',
    'ms_marco',
]
prompt_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/prompts/'
result_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/'
noise_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/noise/'

# if __name__ == '__main__':
#     for dataset_id in datasets:
#         # read prompt
#         prompt_file = f"{prompt_path}{dataset_id}.json"
#         result_file = f"{result_path}{dataset_id}.json"
#         import json
#         with open(prompt_file, 'r') as file:
#             data = json.load(file)
#         target_questions = [item['question'] for item in data]
            
#         # read noise
#         noise_file = f"{noise_path}{dataset_id}_noise.json"
#         with open(noise_file, 'r') as file:
#             noise = json.load(file)
        
#         # sample noise
#         import random
#         noise = random.sample(noise, 250)
#         noise = [item for item in noise if item not in target_questions]
#         noise = random.sample(noise, 200)
        
#         # calculate attacker ppl
#         white_record = []
#         black_record = []
#         for item in data:
#             question = item['question']
#             white = item['white']
#             black = item['black']
            
#             white_response = generate(white)
#             black_response = generate(black)
            
#             white_ppl = perplexity(question + " " + white_response)
#             black_ppl = perplexity(question + " " + black_response)
            
#             white_record.append(
#                 {'value': white_ppl, 'label': 1}
#             )
#             black_record.append(
#                 {'value': black_ppl, 'label': 1}
#             )
            
#         record = white_record + black_record
            
#         with open(result_file, 'w') as file:
#             json.dump(record, file, indent=4)
            
#         # # calculate legit ppl
#         victim_record = []
#         for question in noise:
#             # similar query
#             similar_question = generate_similar_question(question)
#             print(similar_question)
#             if cosine_sim(similar_question, question) >= 0.8 and semantic_score(similar_question, question) >= 0.8:
#                 continue
            
#             # response 
#             print(similar_question)
#             question_response = generate(question)
#             ppl = perplexity(similar_question + " " + question_response)
#             victim_record.append(
#                 {'value': ppl, 'label': 0}
#             )
            
#         with open(result_file, 'a') as file:
#             json.dump(victim_record, file, indent=4)


def get_data():
    # 1. malicious reuse 
    # 2. prompt injection
    # 3. normal reuse 
    # 4. only input and output
    import json
    input_file = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/prompts/squad.json'
    with open(input_file, 'r') as file:
        data = json.load(file)
        
    # each prompt 
    normal_input_output_ppls = []
    normal_reuse_ppls = []
    prompt_injection_ppls = []
    malicious_reuse_ppls = []
    
    for item in data:
        question = item['question']  # normal question
        white = item['white']     # prompt injection and malicious reuse 
        
        white.replace('print', 'introduce')
        
        question_answer = generate(question)
        black_answer = generate(white)
        
        # get similar question to question
        similar_question = generate_similar_question(question)
        
        # ppl
        normal_input_output_ppl = perplexity(question, question_answer)
        normal_reuse_ppl = perplexity(similar_question, question_answer)
        prompt_injection_ppl = perplexity(white, black_answer)
        malicious_reuse_ppl = perplexity(similar_question, black_answer)
        
        # append
        print(f"\noriginal question: {question}")
        print(f"similar question: {similar_question}")
        normal_input_output_ppls.append(normal_input_output_ppl)
        normal_reuse_ppls.append(normal_reuse_ppl)
        prompt_injection_ppls.append(prompt_injection_ppl)
        malicious_reuse_ppls.append(malicious_reuse_ppl)
        
    with open('normal_input_output_ppls.json', 'w') as file:
        json.dump(normal_input_output_ppls, file, indent=4)    
    with open('normal_reuse_ppls.json', 'w') as file:
        json.dump(normal_reuse_ppls, file, indent=4)    
    with open('prompt_injection_ppls.json', 'w') as file:
        json.dump(prompt_injection_ppls, file, indent=4)    
    with open('malicious_reuse_ppls.json', 'w') as file:
        json.dump(malicious_reuse_ppls, file, indent=4)    
        
import json
import matplotlib.pyplot as plt
import seaborn as sns
def analysis():
    # Load data
    with open('normal_input_output_ppls.json', 'r') as file:
        normal_input_output_ppls = json.load(file)
    with open('normal_reuse_ppls.json', 'r') as file:
        normal_reuse_ppls = json.load(file) 
    with open('prompt_injection_ppls.json', 'r') as file:
        prompt_injection_ppls = json.load(file)
    with open('malicious_reuse_ppls.json', 'r') as file:
        malicious_reuse_ppls = json.load(file)

    # Combine for plotting
    data = [
        ('Normal I/O', normal_input_output_ppls),
        ('Normal Reuse', normal_reuse_ppls),
        ('Prompt Injection', prompt_injection_ppls),
        ('Malicious Reuse', malicious_reuse_ppls)
    ]

    # Plot with constraints
    plt.figure(figsize=(10, 6))
    for label, values in data:
        sns.kdeplot(values, label=label, fill=False, clip=(0, None))  # Enforce x >= 0

    plt.xlabel('Perplexity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('PPL Distributions', fontsize=14)
    plt.legend(frameon=False)
    plt.xlim(0, None)  # Ensure x-axis starts at 0
    plt.tight_layout()
    
    # Save before show to avoid blank image
    plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
    plt.show()    
    
def analysis2():
    # Load data in a single dict for efficiency
    files = {
        'normal_reuse_ppls': 'Normal Reuse',
        'malicious_reuse_ppls': 'Malicious Reuse'
    }
    data = {}
    for file, label in files.items():
        with open(f'{file}.json', 'r') as f:
            data[label] = json.load(f)
    
    # Create figure with better styling
    plt.figure(figsize=(10, 6))
    for label, values in data.items():
        plt.hist(values, bins=50, alpha=0.7, label=label, histtype='stepfilled', edgecolor='black')
    
    # Enhance plot
    plt.legend(frameon=False, loc='upper right')
    plt.xlabel('Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of PPL Values', fontsize=14, pad=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
import numpy as np
def analysis3():
    with open('normal_input_output_ppls.json', 'r') as file:
        normal_input_output_ppls = json.load(file)
    with open('normal_reuse_ppls.json', 'r') as file:
        normal_reuse_ppls = json.load(file)
    with open('prompt_injection_ppls.json', 'r') as file:
        prompt_injection_ppls = json.load(file)
    with open('malicious_reuse_ppls.json', 'r') as file:
        malicious_reuse_ppls = json.load(file)

    # Define fixed bins with step size of 10
    bins = np.arange(0, 2000, 10)  # 0, 10, 20, ..., 100

    # Plot histograms with same bins
    # plt.hist(normal_input_output_ppls, bins=bins, alpha=0.5, label='Normal Input Output', rwidth=0.5)
    # plt.hist(normal_reuse_ppls, bins=bins, alpha=0.5, label='Normal Reuse', rwidth=0.5)
    plt.hist(prompt_injection_ppls, bins=bins, alpha=0.5, label='Prompt Injection', rwidth=0.5)
    plt.hist(malicious_reuse_ppls, bins=bins, alpha=0.5, label='Malicious Reuse', rwidth=0.5)        
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')
    plt.legend()
    plt.show()
    plt.savefig('distribution.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    analysis3()