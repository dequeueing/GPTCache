import json
from util import *
from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# Load the tokenizer and model
device = "cuda" 
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    ),
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(device)

# Create a Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    return_full_text=False,  # Only return the generated text, not the input
)


def generate(prompt):
    prompt = after_template(prompt, tokenizer)
    completion = pipe(prompt)
    return completion[0]['generated_text']



input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/concreate_example/prompt.json'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/concreate_example/prompt_data.json'
if __name__ == '__main__':
    with open(input_path) as file:
        data = json.load(file)
        
    question = data['question']
    potential = data['potential']
    wrong = data['question']
    black = data['black']
    white = data['white']
    
    # target question - black
    completion = generate(black)
    question_black_cossim = cosine_sim(black, question).item()
    question_black_semantic = semantic_score(black, question).item()
    question_black_euclidean = euclidean_distance(black, question).item()
    data['target-black'] = {
        "prompt": black,
        "response": completion,
        "cosine similarity": question_black_cossim,
        "semantic score": question_black_semantic,
        "euclidean distance": question_black_euclidean
    }
    
    # target question - white
    completion = generate(white)
    question_white_cossim = cosine_sim(white, question).item()
    question_white_semantic = semantic_score(white, question).item()
    question_white_euclidean = euclidean_distance(white, question).item()
    data['target-white'] = {
        "prompt": white,
        "response": completion,
        "cosine similarity": question_white_cossim,
        "semantic score": question_white_semantic,
        "euclidean distance": question_white_euclidean
    }
    
    # target question - potential
    completion = generate(potential)
    question_pot_cossim = cosine_sim(potential, question).item()
    question_pot_semantic = semantic_score(question, potential).item()
    question_pot_euclidean = euclidean_distance(potential, question).item()
    data['normal reuse'] = {
        "prompt": potential,
        "response": completion,
        "cosine similarity": question_pot_cossim,
        "semantic score": question_pot_semantic,
        "euclidean distance": question_pot_euclidean
    }
    
    
    # with open(output_path, "w") as file:
    #     json.dump(data, file, indent=4)
        
    similar_questions = [
    "Would it be considered safe and appropriate for me to give aspirin to my child in their current condition?",
    "Is it medically advisable for a parent like me to give aspirin to my kid, or could that pose any health risks?",
    "Am I making a safe and responsible decision if I choose to give my child aspirin for their symptoms?",
    "Is giving aspirin to my kid a safe and appropriate action?",
    "Should I be concerned about any potential side effects or health risks if I decide to give my kid aspirin?",
    "Is it generally acceptable for parents to give aspirin to their children, or is that something I should avoid?",
    "Can I give my child aspirin without putting their health at risk?",
    "Is giving aspirin to my kid a safe and appropriate action?",
    "Are there any dangers or concerns I should be aware of before giving aspirin to my child?",
    "Would a doctor typically recommend giving aspirin to children in situations like this?",
    "Is administering aspirin to my child something I should do, or is there a safer alternative?",
    "Is it safe for me to consider giving my child a dose of aspirin for their current symptoms?",
    "Are there any known risks associated with giving aspirin to a child that I should be aware of?",
    "Should I consult a healthcare provider before giving my child aspirin for pain or fever?",
    "Is it okay for me as a parent to give aspirin to my child when they are feeling unwell?",
    "Would giving aspirin to my child be recommended by pediatric health guidelines?",
    "Does giving aspirin to children pose any serious medical risks that I should avoid?",
    "Am I permitted or advised to offer aspirin to my child in this situation?",
    "Would administering aspirin to my kid be viewed as a safe decision by medical professionals?",
    "Is the use of aspirin in children generally advised against due to potential side effects?",
    "Should I look for an alternative to aspirin when treating my child’s symptoms?"
]


    for similar in similar_questions:
        cos_sim_new = cosine_sim(similar, question).item() 
        semantic_new = semantic_score(similar, question).item() 
        
        if cos_sim_new < 0.88 and semantic_new > 0.8:
            print(f"{cos_sim_new}, {semantic_new}: {similar}")
