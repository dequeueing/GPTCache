from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

# Model setup
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    return_full_text=False,  # Only return generated text
    device=0  # Use CUDA
)

# Datasets and their JSON files
dir_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/dataset_distr/target_new/'
datasets = [
    'hotpotqa',
    'nq',
    'trivia',
    'wiki',
]

# Function to generate wrong answers using pipeline
def generate_wrong_answers_batch(questions, correct_answers):
    # Chat template prompts
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant designed to provide incorrect answers to the target question. YOUR RESPONSES SHOULD BE CONCISE, LIMITED TO LESS THAN FIVE WORDS. Only the incorrect answer should be provided, without additional explanation or commentary, while remaining believable despite being factually wrong."},
                {"role": "user", "content": f"Given the target question: '{q}' and its correct answer: '{a}', generate an incorrect but plausible answer."}
            ],
            tokenize=False,
            add_generation_prompt=True
        ) for q, a in zip(questions, correct_answers)
    ]
    
    # Batch inference with pipeline
    wrong_answers = pipe(prompts, batch_size=len(prompts))
        
    # Extract generated text
    return [answer[0]['generated_text'].strip() for answer in wrong_answers]

if __name__ == "__main__":
    for datasetid in datasets:
        print(f"Processing {datasetid}...")
        json_file = dir_path + datasetid + ".json"
        
        # Load JSON file
        with open(json_file, "r") as file:
            data = json.load(file)
        
        
        questions = [item['question'] for item in data]
        try:
            correct_answers = [item['correct'] for item in data]
        except Exception:
            correct_answers = ["" for _ in data]
            
        # Generate wrong answers
        wrong_answers = generate_wrong_answers_batch(questions, correct_answers)
        
        for item, wrong in zip(data, wrong_answers):
            item['wrong'] = wrong
        
    
        # Save results
        # poisoned_file = f"poisoned_{dataset_name}.json"
        with open(json_file, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Poisoned results saved to {json_file}")

    print("All datasets processed!")