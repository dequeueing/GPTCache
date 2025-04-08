import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import pipeline
from testing import *



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
        system_prompt = "You are a helpful assistant that generates OEN semantically identical question based on the user's input. No additional information should be output. "
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

def perplexity(text: str) -> float:
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

if __name__ == '__main__':
    for dataset_id in datasets:
        # read prompt
        prompt_file = f"{prompt_path}{dataset_id}.json"
        result_file = f"{result_path}{dataset_id}.json"
        import json
        with open(prompt_file, 'r') as file:
            data = json.load(file)
        target_questions = [item['question'] for item in data]
            
        # read noise
        noise_file = f"{noise_path}{dataset_id}_noise.json"
        with open(noise_file, 'r') as file:
            noise = json.load(file)
        
        # sample noise
        import random
        noise = random.sample(noise, 250)
        noise = [item for item in noise if item not in target_questions]
        noise = random.sample(noise, 200)
        
        # # calculate attacker ppl
        # white_record = []
        # black_record = []
        # for item in data:
        #     question = item['question']
        #     white = item['white']
        #     black = item['black']
            
        #     white_response = generate(white)
        #     black_response = generate(black)
            
        #     white_ppl = perplexity(question + " " + white_response)
        #     black_ppl = perplexity(question + " " + black_response)
            
        #     white_record.append(
        #         {'value': white_ppl, 'label': 1}
        #     )
        #     black_record.append(
        #         {'value': black_ppl, 'label': 1}
        #     )
            
        # record = white_record + black_record
            
        # with open(result_file, 'w') as file:
        #     json.dump(record, file, indent=4)
            
        # # calculate legit ppl
        victim_record = []
        for question in noise:
            # similar query
            similar_question = generate_similar_question(question)
            print(similar_question)
            if cosine_sim(similar_question, question) >= 0.8 and semantic_score(similar_question, question) >= 0.8:
                continue
            
            # response 
            print(similar_question)
            question_response = generate(question)
            ppl = perplexity(similar_question + " " + question_response)
            victim_record.append(
                {'value': ppl, 'label': 0}
            )
            
        with open(result_file, 'a') as file:
            json.dump(victim_record, file, indent=4)