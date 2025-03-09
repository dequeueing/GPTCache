import logging
import torch

logger = logging.getLogger()


def post_proc(token_embeddings, inputs):
    """convert token embedding to sentence embedding, mean the dimensions across all tokens"""
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    input_mask_expanded = input_mask_expanded.to(token_embeddings.device)
    sentence_embs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embs

def cosine_sim(model, tokenizer, query1, query2):
    query1_tokenized = tokenizer(query1, return_tensors="pt", padding=False).to(model.device)
    query2_tokenized = tokenizer(query2, return_tensors="pt", padding=False).to(model.device)
    q1_embedding = model(**query1_tokenized).last_hidden_state
    q2_embedding = model(**query2_tokenized).last_hidden_state
    q1_sent_emb = post_proc(q1_embedding, query1_tokenized).squeeze(0)
    q2_sent_emb = post_proc(q2_embedding, query2_tokenized).squeeze(0)
    return torch.nn.CosineSimilarity(dim=0)(
        q1_sent_emb, q2_sent_emb
    )
    


def embedding_token_gradient(
    embedding_model,
    embedding_tokenizer,
    attacker_prompt_ids_dict,
    control_start,
    control_end,
    victim_query,
    device,
):
    """Return a gradient tensor of the embedding suffix."""
    logger.debug("\n===========token_gradients===============")
    # Get the victim sentence embedding to avoid backward error.
    victim_input_ids = embedding_tokenizer(
        victim_query, return_tensors="pt", padding=True
    ).to(device)
    victim_embedding = embedding_model(**victim_input_ids).last_hidden_state
    victim_sentence_embedding = post_proc(victim_embedding, victim_input_ids).squeeze(0)

    # get the embedding layer of bert
    embedding = embedding_model.get_input_embeddings()
    logger.debug(f"{embedding.weight.size()}")

    # get token
    token_id = attacker_prompt_ids_dict["input_ids"]
    attention_mask = attacker_prompt_ids_dict["attention_mask"]
    control_token_ids = token_id[0][control_start : control_end + 1]
    logger.debug(f"{control_token_ids}")

    # get onehot for the control tokens
    control_slice_len = control_end - control_start + 1
    one_hot = torch.zeros(
        control_slice_len, embedding.weight.size(0), device=embedding_model.device
    )
    control_token_pos = torch.arange(control_slice_len)
    one_hot[control_token_pos, control_token_ids] = 1
    logger.debug(f"one hot size: {one_hot.size()}")

    # set requires_grad to True
    one_hot.requires_grad = True

    # get input embedding to be forwarded
    input_embed = (one_hot @ embedding.weight).unsqueeze(0)
    attacker_tokenized = attacker_prompt_ids_dict
    attacker_embedding = embedding.weight[attacker_tokenized["input_ids"]]

    logger.debug(f"attacker tokens ids: : {attacker_tokenized['input_ids']}")
    logger.debug(f"attacker embedding: {attacker_embedding.size()}")

    # replace control tokens
    attacker_embedding_clone = attacker_embedding.clone()
    attacker_embedding_clone[:, control_start : control_end + 1, :] = input_embed
    attacker_embedding_clone.to(device)
    logger.debug(f"replaced embedding: {attacker_embedding_clone.size()}")
    
    # check attacker embedding_clone same as attacker embedding
    assert torch.equal(attacker_embedding_clone, attacker_embedding)
    logger.debug("replaced tensor same as the previous ones.")

    def get_sentence_embedding(raw_model, attacker_embedding, attacker_tokenized):
        # forward!
        model_output = raw_model(
            inputs_embeds=attacker_embedding,
            attention_mask=attacker_tokenized["attention_mask"],
        ).last_hidden_state
        sentence_embedding = post_proc(model_output, attacker_tokenized).squeeze(0)
        return sentence_embedding

    # forward!
    embedding_model.eval()
    attacker_sentence_embedding = get_sentence_embedding(
        embedding_model, attacker_embedding_clone, attacker_tokenized
    )
    logger.debug(
        f"type of attacker sentence embedding: {type(attacker_sentence_embedding)}"
    )
    logger.debug(
        f"shape of attacker sentence embedding: {attacker_sentence_embedding.size()}"
    )

    # compute the cosine sim btw attacker and victim
    cos_sim = torch.nn.CosineSimilarity(dim=0)(
        attacker_sentence_embedding, victim_sentence_embedding
    )
    logger.debug(f"cosine similarity: {cos_sim}")

    # backward!
    cos_sim.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    logger.debug(f"grad: {grad.size()}")
    logger.debug(f"grad: {grad[0]}")

    logger.debug("===========token_gradients===============\n")
    return grad

def embedding_sample_control(adv_control_tokens, coordinate_gradient, batch_size, topk):
    logging.debug("\n===========sample_control===============")
    
    top_indices = coordinate_gradient.topk(topk, dim=1).indices
    adv_control_tokens = adv_control_tokens.to(coordinate_gradient.device)
    logging.debug(f"shape of top indices: {top_indices.shape}")
    logging.debug(f"adv: {adv_control_tokens}")
    
    original_control_tokens = adv_control_tokens.repeat(batch_size, 1)
    logging.debug(f"shape of original_control_tokens: {original_control_tokens.shape}")
    
    logging.debug(f"the len: {len(adv_control_tokens[0])}")
    new_token_pos = torch.arange(
        0,
        len(adv_control_tokens[0]),
        len(adv_control_tokens[0])/batch_size,
        device=coordinate_gradient.device
    ).type(torch.int64)
    
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1), device=coordinate_gradient.device),
    )
    new_control_tokens = original_control_tokens.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    logging.debug(f"the new control tokens: {new_control_tokens}")
    
    logging.debug("===========sample_control===============\n")
    return new_control_tokens


def emb_get_filtered_candidates(tokenizer, control_candidates, current_control):
    logging.debug("\n==========get_filtered_candidates===============")
    logging.debug(f"current_control: {current_control}")
    logging.debug(f"new_adv_suffix_toks: {control_candidates}")
    
    cands = []
    for i in range(control_candidates.shape[0]):
        decoded_str = tokenizer.decode(control_candidates[i])
        if decoded_str != current_control and len(tokenizer(decoded_str)['input_ids']) == len(control_candidates[i]) + 2:
            cands.append(decoded_str)
    
    cands = cands + [cands[-1]] * (len(control_candidates) - len(cands))
    
    logging.debug("===========get_filtered_candidates===============\n")
    return cands


def emb_get_logits(model, tokenizer, suffix_semantic, control_start, control_end, test_controls, attacker_base, victim_query):
    logging.debug("\n==========get_logits===============")
    
    logging.debug(f"control_start:{control_start}, control_end:{control_end}")
    logging.debug(f"first of test controls: {test_controls[0]}, last of test controls: {test_controls[-1]}")
    
    def get_sent_emb_from_text(text):
        input_ids = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        embedding = model(**input_ids).last_hidden_state
        sentence_embedding = post_proc(embedding, input_ids).squeeze(0)
        return sentence_embedding
    
    max_score = 0
    best_suffix = None
    for i in range(len(test_controls)):
        this_suffix = test_controls[i]
        new_attacker_query = attacker_base + this_suffix + suffix_semantic
        new_attacker_sen_emb = get_sent_emb_from_text(new_attacker_query)
        victim_sentence_embedding = get_sent_emb_from_text(victim_query)
        score = torch.nn.CosineSimilarity(dim=0)(new_attacker_sen_emb, victim_sentence_embedding)
        if score > max_score:
            max_score = score
            best_suffix = this_suffix
            
    # logging.debug(f"suffix {best_suffix} has best score: {max_score}")
        
    logging.debug("==========get_logits===============\n")
    return max_score, best_suffix

