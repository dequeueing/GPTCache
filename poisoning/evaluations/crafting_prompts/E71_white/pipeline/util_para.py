import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.cross_encoder import CrossEncoder


num_iter = 100
threshold_target_function = 0.8

def set_seed():
    temp = 30
    np.random.seed(temp)
    torch.manual_seed(temp)
    torch.cuda.manual_seed_all(temp)


def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',  # Optional: to include timestamps and log levels
        filename='semantic.log',  # Specify the file where logs should be saved
        filemode='w'  # 'a'  append, 'w' overwrite
    )
    

device = 'cuda'
embedding_model_type = "distilbert-base-uncased"
semantic_model_type = "cross-encoder/quora-distilroberta-base"

embedding_model = AutoModel.from_pretrained(embedding_model_type).to(device)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_type)

semantic_encoder = CrossEncoder(semantic_model_type)
semantic_model = semantic_encoder.model.to(device)
semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_type)


def semantic_score(q1, q2):
    score = semantic_encoder.predict([(q1, q2)], show_progress_bar=False)[0]
    return score


def _sent_embed_from_hidden(hidden_states, attention_mask):
    """convert token embedding to sentence embedding, mean the dimensions across all tokens"""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )
    input_mask_expanded = input_mask_expanded.to(hidden_states.device)
    sentence_embs = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embs.squeeze()


def _sent_embed_from_text(text):
    # Tokenization
    text_tokenized = embedding_tokenizer(text, return_tensors="pt", padding=False).to(
        embedding_model.device
    )
    hidden_states_embedding = embedding_model(**text_tokenized).last_hidden_state
    return _sent_embed_from_hidden(
        hidden_states_embedding, text_tokenized["attention_mask"]
    )

    
    
def cosine_sim(query1, query2):
    # Tokenization
    query1_tokenized = embedding_tokenizer(
        query1, return_tensors="pt", padding=False
    ).to(embedding_model.device)
    query2_tokenized = embedding_tokenizer(
        query2, return_tensors="pt", padding=False
    ).to(embedding_model.device)

    # Forward
    q1_embedding = embedding_model(**query1_tokenized).last_hidden_state
    q2_embedding = embedding_model(**query2_tokenized).last_hidden_state

    # Get sentence embedding from layer activation
    q1_sent_emb = _sent_embed_from_hidden(
        q1_embedding, query1_tokenized["attention_mask"]
    )
    q2_sent_emb = _sent_embed_from_hidden(
        q2_embedding, query2_tokenized["attention_mask"]
    )

    # Cosine similarity
    return torch.nn.CosineSimilarity(dim=0)(q1_sent_emb, q2_sent_emb)

    
def find_embedding_suffix_indices(attacker: str, embedding_suffix: str, semantic_suffix: str, base: str) -> list[int]:
    # Find start1 and end1
    tokenizer = embedding_tokenizer
    attacker_tokenized = tokenizer(
        attacker, return_tensors="pt", padding=True, return_offsets_mapping=True
    )
    emb_suffix_tokenized = tokenizer(
        embedding_suffix, return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=False
    )
    sem_suffix_tokenized = tokenizer(
        semantic_suffix, return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=False
    )
    base_tokenized = tokenizer(
        base, return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=False
    )

    
    attacker_ids = attacker_tokenized["input_ids"][0].tolist()
    sem_suffix_ids = sem_suffix_tokenized["input_ids"][0].tolist()
    emb_suffix_ids = emb_suffix_tokenized["input_ids"][0].tolist()
    base_ids = base_tokenized["input_ids"][0].tolist()
    sem_len = len(sem_suffix_ids)
    emb_len = len(emb_suffix_ids)
    base_len = len(base_ids)
    
    start1 = base_len + 1 
    end1 = base_len + 1 + emb_len
    return start1, end1


def get_input_tokenized_embedding(query_text, special_tokens=True):
    return embedding_tokenizer(
        query_text, return_tensors="pt", padding=True, add_special_tokens=special_tokens
    )

def get_input_tokenized_semantic(query_text, special_tokens=True):
    return semantic_tokenizer(
        query_text, return_tensors="pt", padding=True, add_special_tokens=special_tokens
    )


def embed_token_gradient(
    attacker_ids,
    attacker_attention_mask,
    control_start,
    control_end,
    victim_query,
):
    """Return a gradient tensor of the embedding suffix."""
    logging.debug("\n===========token_gradients===============")
    # Get the victim sentence embedding to avoid backward error.
    victim_sentence_embedding = _sent_embed_from_text(victim_query)

    # get the embedding layer of bert
    embedding = embedding_model.get_input_embeddings()
    logging.debug(f"{embedding.weight.size()}")

    # get token
    control_token_ids = attacker_ids[0][control_start : control_end + 1]

    # get onehot for the control tokens
    control_slice_len = control_end - control_start + 1
    one_hot = torch.zeros(
        control_slice_len, embedding.weight.size(0), device=embedding_model.device
    )
    control_token_pos = torch.arange(control_slice_len)
    one_hot[control_token_pos, control_token_ids] = 1

    logging.debug(f"{control_token_ids}")
    logging.debug(f"one hot size: {one_hot.size()}")

    # set requires_grad to True
    one_hot.requires_grad = True

    # get input embedding to be forwarded
    control_embed = (one_hot @ embedding.weight).unsqueeze(0)
    attacker_embedding = embedding.weight[attacker_ids]

    logging.debug(f"attacker tokens ids: : {attacker_ids}")
    logging.debug(f"attacker embedding: {attacker_embedding.size()}")

    # replace control tokens
    attacker_embedding_clone = attacker_embedding.clone()
    attacker_embedding_clone[:, control_start : control_end + 1, :] = control_embed
    attacker_embedding_clone = attacker_embedding_clone.to(control_embed.device)

    # check attacker embedding_clone same as attacker embedding
    assert torch.equal(attacker_embedding_clone, attacker_embedding)
    logging.debug("replaced tensor same as the previous ones.")

    def _sent_embed_from_input_embeddding(input_embedding, attention_mask):
        hidden_states = embedding_model(
            inputs_embeds=input_embedding, attention_mask=attention_mask
        ).last_hidden_state
        return _sent_embed_from_hidden(hidden_states, attention_mask)

    # forward!
    embedding_model.eval()
    # attacker_sentence_embedding = get_sentence_embedding(
    #     embedding_model, attacker_embedding_clone, attacker_tokenized
    # )
    attacker_sentence_embedding = _sent_embed_from_input_embeddding(
        attacker_embedding_clone, attacker_attention_mask
    )
    logging.debug(
        f"type of attacker sentence embedding: {type(attacker_sentence_embedding)}"
    )
    logging.debug(
        f"shape of attacker sentence embedding: {attacker_sentence_embedding.size()}"
    )

    # compute the cosine sim btw attacker and victim
    cos_sim = torch.nn.CosineSimilarity(dim=0)(
        attacker_sentence_embedding.squeeze(), victim_sentence_embedding.squeeze()
    )
    # backward!
    cos_sim.backward()
        

    # get gradient
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    logging.debug(f"grad: {grad.size()}")
    logging.debug(f"grad: {grad[0]}")

    logging.debug("===========token_gradients===============\n")
    return grad


def sample_control(control_tokens, coordinate_gradient, batch_size, topk):
    logging.debug("\n===========sample_control===============")

    top_indices = coordinate_gradient.topk(topk, dim=1).indices
    control_tokens = control_tokens.to(coordinate_gradient.device)
    logging.debug(f"shape of top indices: {top_indices.shape}")
    logging.debug(f"adv: {control_tokens}")

    original_control_tokens = control_tokens.repeat(batch_size, 1)
    logging.debug(f"shape of original_control_tokens: {original_control_tokens.shape}")

    logging.debug(f"the len: {len(control_tokens[0])}")
    new_token_pos = torch.arange(
        0,
        len(control_tokens[0]),
        len(control_tokens[0]) / batch_size,
        device=coordinate_gradient.device,
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=coordinate_gradient.device),
    )
    new_control_tokens = original_control_tokens.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )
    logging.debug(f"the new control tokens: {new_control_tokens}")

    logging.debug("===========sample_control===============\n")
    return new_control_tokens


def emb_get_filtered_candidates_embedding_only(
    control_candidates, current_control, current_query, base, suffix_sem, device="cuda"
):
    logging.debug("\n==========get_filtered_candidates===============")
    logging.debug(f"current_control: {current_control}")
    logging.debug(f"new_adv_suffix_toks: {control_candidates}")
    logging.debug(f"current_query: {current_query}")

    # Ensure tokenizer is defined globally or passed as an argument
    global embedding_tokenizer  # Replace with your tokenizer if different
    # if not isinstance(embedding_tokenizer, AutoTokenizer):
    #     print(f"type: {embedding_tokenizer}")
    #     raise ValueError("embedding_tokenizer must be a valid tokenizer instance")

    # Move control_candidates to CUDA if it's a tensor
    if isinstance(control_candidates, torch.Tensor):
        control_candidates = control_candidates.to(device)
    else:
        control_candidates = torch.tensor(control_candidates, device=device)

    # Get the target length from current_query
    current_query_len = len(embedding_tokenizer(current_query, add_special_tokens=False)["input_ids"])

    # Decode all candidates in batch
    decoded_strs = embedding_tokenizer.batch_decode(
        control_candidates, skip_special_tokens=True
    )

    # Construct new queries in batch (list comprehension for now, could be tensorized further)
    new_queries = [base + decoded_str + suffix_sem for decoded_str in decoded_strs]

    # Tokenize all new_queries in a single batch
    tokenized_batch = embedding_tokenizer(
        new_queries,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,  # Pad to longest in batch
        truncation=False
    )

    # Move tokenized data to CUDA
    input_ids = tokenized_batch["input_ids"].to(device)

    # Compute lengths of tokenized sequences (number of non-padding tokens per sequence)
    lengths = torch.sum(input_ids != embedding_tokenizer.pad_token_id, dim=1)

    # Filter candidates where length matches current_query_len
    mask = lengths == current_query_len
    filtered_indices = torch.where(mask)[0]

    # Extract filtered candidates
    if len(filtered_indices) == 0:
        logging.warning("Only one candidate after filter.")
        return [current_control]  # Fallback to current_control

    # Convert filtered indices back to decoded strings
    cands = [decoded_strs[i.item()] for i in filtered_indices]

    logging.debug("===========get_filtered_candidates===============\n")
    return cands

def find_semantic_suffix_indices(attacker: str, embedding_suffix: str, semantic_suffix: str, base: str):
    # Initialize model and tokenizer
    model_id = "cross-encoder/quora-distilroberta-base"
    encoder = CrossEncoder(model_id)
    tokenizer = encoder.tokenizer

    # Tokenize attacker
    attacker_tokenized = tokenizer(
        attacker, return_tensors="pt", padding=True, return_offsets_mapping=True
    )

    # Tokenize suffix with a leading space to match attacker's context
    sem_suffix_tokenized = tokenizer(
        semantic_suffix, return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=False
    )
    
    # TODO: change the hardcore here.
    target_token = 33130
    attacker_ids = attacker_tokenized["input_ids"][0].tolist()
    victim_ids = sem_suffix_tokenized["input_ids"][0].tolist()

    start2 = None
    end2 = None    
    for i, item in reversed(list(enumerate(attacker_ids))):
        if item == target_token:
            if not end2:
                end2 = i
            else:
                start2 = i
        elif start2 and end2:
            break
    
    return start2, end2 



def emb_get_logits(
    suffix_semantic,
    control_start,
    control_end,
    test_controls,
    attacker_base,
    victim_query,
):
    """Batch processing"""
    logging.debug("\n==========get_logits===============")
    logging.debug(f"control_start:{control_start}, control_end:{control_end}")
    logging.debug(
        f"first of test controls: {test_controls[0]}, last of test controls: {test_controls[-1]}"
    )

    # Combine attacker_base with each test_control and suffix_semantic
    new_attacker_queries = [attacker_base + ctrl + suffix_semantic for ctrl in test_controls]
    
    # Get embeddings for all attacker queries in a batch and move to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_attacker_sen_embs = torch.stack(
        [_sent_embed_from_text(query) for query in new_attacker_queries]
    ).to(device)  # Shape: (batch_size, embedding_dim)
    
    # Get victim embedding once and move to CUDA
    victim_sentence_embedding = _sent_embed_from_text(victim_query).unsqueeze(0).to(device)  # Shape: (1, embedding_dim)

    # Compute cosine similarity for all embeddings in batch on CUDA
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    scores = cos_sim(new_attacker_sen_embs, victim_sentence_embedding.expand_as(new_attacker_sen_embs))

    # Find the best score and corresponding suffix
    max_score, best_idx = torch.max(scores, dim=0)
    best_suffix = test_controls[best_idx.item()]

    logging.debug("==========get_logits===============\n")
    return max_score.item(), best_suffix


def sem_token_gradients_penalty2(
    attacker_ids,
    attacker_mask,
    control_start,
    control_end,
    attacker_query,
    victim_query,
    original_cosine_sim=0.9,  # Add as argument
    lambda_penalty=5.0,         # Strong penalty weight
):
    logging.debug("\n===========token_gradients===============")

    semantic_embedding = semantic_model.get_input_embeddings()
    embedding_embedding = embedding_model.get_input_embeddings()

    # Semantic one-hot
    control_token_ids_sem = attacker_ids[0][control_start : control_end + 1]
    control_slice_len = control_end - control_start + 1
    one_hot_sem = torch.zeros(
        control_slice_len, semantic_embedding.weight.size(0), device=semantic_model.device
    )
    control_token_pos = torch.arange(control_slice_len)
    one_hot_sem[control_token_pos, control_token_ids_sem] = 1
    one_hot_sem.requires_grad_()

    input_embed_sem = (one_hot_sem @ semantic_embedding.weight).unsqueeze(0)

    ### SBERT Score ###
    sentences = [(attacker_query, victim_query)]
    input_tokenized = semantic_tokenizer(
        sentences, return_tensors="pt", padding=True
    ).to(semantic_model.device)

    input_embedding_eg = semantic_embedding.weight[input_tokenized["input_ids"]]
    input_embedding_eg_dup = input_embedding_eg.clone()
    input_embedding_eg_dup[:, control_start : control_end + 1, :] = input_embed_sem

    semantic_model.eval()
    model_predictions = semantic_model(
        inputs_embeds=input_embedding_eg_dup,
        attention_mask=input_tokenized["attention_mask"],
        return_dict=True,
    )
    logits = torch.nn.Sigmoid()(model_predictions.logits)
    pred_score = logits[0][0]
    logging.debug(f"SBERT prediction: {pred_score}")

    ### Cosine Similarity ###
    attacker_tokenized_emb = embedding_tokenizer(
        attacker_query, return_tensors="pt", padding=False
    ).to(embedding_model.device)
    victim_tokenized_emb = embedding_tokenizer(
        victim_query, return_tensors="pt", padding=False
    ).to(embedding_model.device)

    emb_input_ids = attacker_tokenized_emb["input_ids"][0]
    control_start_emb = len(emb_input_ids) - control_slice_len
    control_end_emb = len(emb_input_ids) - 1
    control_token_ids_emb = emb_input_ids[control_start_emb : control_end_emb + 1]

    one_hot_emb = torch.zeros(
        control_slice_len, embedding_embedding.weight.size(0), device=embedding_model.device
    )
    one_hot_emb[control_token_pos, control_token_ids_emb] = 1
    one_hot_emb.requires_grad_()

    input_embed_emb = (one_hot_emb @ embedding_embedding.weight).unsqueeze(0)

    attacker_input_emb = embedding_embedding.weight[attacker_tokenized_emb["input_ids"]]
    attacker_input_emb_dup = attacker_input_emb.clone()
    attacker_input_emb_dup[:, control_start_emb : control_end_emb + 1, :] = input_embed_emb

    attacker_hidden = embedding_model(
        inputs_embeds=attacker_input_emb_dup,
        attention_mask=attacker_tokenized_emb["attention_mask"]
    ).last_hidden_state
    attacker_sent_emb = _sent_embed_from_hidden(
        attacker_hidden, attacker_tokenized_emb["attention_mask"]
    )

    victim_input_emb = embedding_embedding.weight[victim_tokenized_emb["input_ids"]]
    victim_hidden = embedding_model(
        inputs_embeds=victim_input_emb,
        attention_mask=victim_tokenized_emb["attention_mask"]
    ).last_hidden_state
    victim_sent_emb = _sent_embed_from_hidden(
        victim_hidden, victim_tokenized_emb["attention_mask"]
    )

    cosine_similarity = torch.nn.CosineSimilarity(dim=0)(attacker_sent_emb, victim_sent_emb)
    logging.debug(f"Cosine similarity: {cosine_similarity}")

    ### Combine Objectives with Penalty ###
    penalty = torch.max(torch.tensor(0.0, device=semantic_model.device), original_cosine_sim - cosine_similarity)
    total_score = pred_score - lambda_penalty * penalty.pow(2)
    logging.debug(f"Penalty: {penalty}, Total score: {total_score}")

    ### Backward Pass ###
    total_score.backward()
    grad = one_hot_sem.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    logging.debug("===========token_gradients===============\n")
    return grad


def sem_get_filtered_candidates(
    control_candidates, current_control, current_query, base, suffix_embed
):
    logging.debug("\n==========sample_control===============")
    logging.debug(f"current_control: {current_control}")
    logging.debug(f"new_adv_suffix_toks: {control_candidates}")

    current_query_len = len(semantic_tokenizer(current_query, add_special_tokens=False)['input_ids'])
    
    logging.debug(f"current_query: {current_query}")
    logging.debug(f"the input ids: {semantic_tokenizer(current_query, add_special_tokens=False)['input_ids']}")
    logging.debug(f"current_query_len: {current_query_len}")

    cands = []
    for i in range(control_candidates.shape[0]):
        decoded_str = semantic_tokenizer.decode(control_candidates[i])
        new_query = base + suffix_embed + decoded_str
        if (len(semantic_tokenizer(new_query, add_special_tokens=False)["input_ids"]) == current_query_len):
            cands.append(decoded_str)

    if len(cands) == 0:
        cands.append(current_control)
        return cands

    logging.debug("===========sample_control===============\n")
    return cands


def sem_get_logits(
    control_start,
    control_end,
    test_controls,
    attacker_base,
    victim_query,
    embedding_suffix,
):
    logging.debug("\n==========get_logits===============")
    logging.debug(f"control_start:{control_start}, control_end:{control_end}")
    logging.debug(
        f"first of test controls: {test_controls[0]}, last of test controls: {test_controls[-1]}"
    )

    # Prepare batch of attacker queries
    # attacker_queries = [attacker_base + embedding_suffix + suffix for suffix in test_controls]
    attacker_queries = [suffix + attacker_base for suffix in test_controls]
    victim_queries = [victim_query] * len(test_controls)  # Repeat victim_query for each test_control

    # Batch prediction
    scores = semantic_encoder.predict(
        list(zip(victim_queries, attacker_queries)),
        show_progress_bar=False,
        batch_size=32 
    )

    # Find the best score and corresponding suffix
    max_score = max(scores)
    best_suffix_idx = scores.argmax()
    best_suffix = test_controls[best_suffix_idx]

    logging.debug(f"suffix {best_suffix} has best score: {max_score}")
    logging.debug("==========get_logits===============\n")
    return max_score, best_suffix
