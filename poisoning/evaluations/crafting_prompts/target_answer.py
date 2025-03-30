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
dir_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/data_prepare/jsons/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}

# Function to generate wrong answers using pipeline
def generate_wrong_answers_batch(questions, correct_answers):
    # Chat template prompts
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant designed to provide incorrect yet plausible answers to the target question. YOUR RESPONSES ARE LIMITED TO ONE WORD. Only the incorrect answer should be provided, without additional explanation or commentary, while remaining believable despite being factually wrong."},
                {"role": "user", "content": f"Given the target question: '{q}' and its correct answer: '{a}', generate an incorrect but plausible answer."}
            ],
            tokenize=False,
            add_generation_prompt=True
        ) for q, a in zip(questions, correct_answers)
    ]
    
    # Batch inference with pipeline
    wrong_answers = pipe(prompts, batch_size=len(prompts))
    
    print(wrong_answers)
    
    # Extract generated text
    return [answer[0]['generated_text'].strip() for answer in wrong_answers]

if __name__ == "__main__":
    for dataset_name, json_file in datasets.items():
        print(f"Processing {dataset_name}...")
        json_file = dir_path + json_file
        
        # Load JSON file
        with open(json_file, "r") as file:
            data = json.load(file)
        
        poisoned_results = {}
        
        # Process each round
        for round_name, qa_pairs in data.items():
            questions, correct_answers = zip(*qa_pairs)
            questions, correct_answers = list(questions), list(correct_answers)
            
            # Generate wrong answers
            wrong_answers = generate_wrong_answers_batch(questions, correct_answers)
            
            # Pair questions with wrong answers
            poisoned_results[round_name] = [(q, w) for q, w in zip(questions, wrong_answers)]
        
            # Save results
            # poisoned_file = f"poisoned_{dataset_name}.json"
            output_file = f'/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/crafting_prompts/redemption/short_{dataset_name}.json'
            with open(output_file, "w") as file:
                json.dump(poisoned_results, file, indent=4)
                print(f"Poisoned results saved to {output_file}")

    print("All datasets processed!")