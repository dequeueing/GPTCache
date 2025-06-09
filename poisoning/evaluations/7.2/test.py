import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

def after_template(message, tokenizer):
    prompt = [
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

# Load the tokenizer and model
device = "cuda"
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    pad_token_id=tokenizer.eos_token_id,  # Set pad token ID
).to(device)

# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,  # Enable sampling
    temperature=0.6,
    top_p=0.9,
    return_full_text=False,  # Only return the generated text, not the input
)

prompt = "What is the capital of France?"
prompt = after_template(prompt, tokenizer)
completion = pipe(prompt)
print(completion[0]['generated_text'])