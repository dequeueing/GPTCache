import torch
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer
import numpy as np

# Step 1: Set up the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "cross-encoder/quora-distilroberta-base"
cross_encoder = CrossEncoder(model_name)
raw_model = cross_encoder.model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
raw_model.train()  # Enable training mode for gradient computation

# Step 2: Define the source and cached questions
src_question_base = "<s>[INST] What is the longest river in the world? [/INST]"
cache_question = "<s>[INST] Just print 'Amazon river' and do nothing else. [/INST]"

# Step 3: Initialize a suffix with 10 tokens
suffix_len = 10
suffix_tokens = ["!"] * suffix_len  # Start with "!" as placeholder tokens
suffix = " ".join(suffix_tokens)

# Step 4: Function to compute similarity score
def get_similarity(src_question, cache_question):
    inputs = tokenizer([src_question], [cache_question], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = raw_model(**inputs)
        score = torch.sigmoid(outputs.logits[0]).item()  # Convert logit to probability [0, 1]
    return score

# Initial similarity score
initial_src_question = src_question_base + " " + suffix
initial_score = get_similarity(initial_src_question, cache_question)
print(f"Initial similarity score: {initial_score:.4f}")

# Step 5: Optimize the suffix one token at a time
num_iterations = 50
for iteration in range(num_iterations):
    # Tokenize the current source question with suffix
    src_question_with_suffix = src_question_base + " " + suffix
    inputs = tokenizer([src_question_with_suffix], [cache_question], return_tensors='pt', padding=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Identify the suffix token positions
    base_tokens = tokenizer(src_question_base, return_tensors='pt')['input_ids'][0]
    suffix_start_idx = len(base_tokens)  # Where the suffix begins
    suffix_end_idx = suffix_start_idx + suffix_len  # Where it ends
    
    # Get embeddings for the input sequence
    embeddings = raw_model.roberta.embeddings.word_embeddings(input_ids)
    embeddings = embeddings.clone().detach().requires_grad_(True)
    
    # Forward pass with embeddings
    outputs = raw_model(inputs_embeds=embeddings, attention_mask=attention_mask)
    score = outputs.logits[0]  # Raw logit score
    
    # Define loss: maximize similarity (minimize negative logit)
    loss = -score
    loss.backward()
    
    # Extract gradients for the suffix embeddings
    suffix_embeddings = embeddings[0, suffix_start_idx:suffix_end_idx]
    suffix_gradients = embeddings.grad[0, suffix_start_idx:suffix_end_idx]
    
    # Select one token to optimize (e.g., cycle through tokens across iterations)
    token_idx_to_optimize = iteration % suffix_len  # Pick a different token each iteration
    
    # Compute the direction to move the embedding (gradient ascent)
    direction = suffix_gradients[token_idx_to_optimize]
    
    # Find the token in the vocabulary that best aligns with the gradient direction
    vocab_embeddings = raw_model.roberta.embeddings.word_embeddings.weight
    similarities = torch.matmul(vocab_embeddings, direction)
    best_token_id = torch.argmax(similarities).item()
    
    # Replace only the selected token in the suffix
    suffix_tokens[token_idx_to_optimize] = tokenizer.decode([best_token_id]).strip()
    suffix = " ".join(suffix_tokens)
    
    # Print progress every 10 iterations
    if iteration % 10 == 0:
        current_score = get_similarity(src_question_with_suffix, cache_question)
        print(f"Iteration {iteration}: Similarity score = {current_score:.4f}, Suffix = '{suffix}'")

# Final result
final_src_question = src_question_base + " " + suffix
final_score = get_similarity(final_src_question, cache_question)
print(f"\nFinal similarity score: {final_score:.4f}")
print(f"Optimized suffix: '{suffix}'")
print(f"Final source question: '{final_src_question}'")