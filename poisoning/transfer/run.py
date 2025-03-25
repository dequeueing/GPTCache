import logging
import torch
import json
import time
import numpy as np
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sentence_transformers.cross_encoder import CrossEncoder


def set_seed():
    np.random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)


def set_logging():
    logging.basicConfig(level=logging.INFO)


def get_input_tokenized_semantic(query_text, special_tokens=True):
    return semantic_tokenizer(
        query_text, return_tensors="pt", padding=True, add_special_tokens=special_tokens
    )


def get_input_tokenized_embedding(query_text, special_tokens=True):
    return embedding_tokenizer(
        query_text, return_tensors="pt", padding=True, add_special_tokens=special_tokens
    )


def suffix_slice(total_string, substring):
    """Get control slice for embedding and semantic suffix"""
    # check shape
    if total_string.dim() != 1 or substring.dim() != 1:
        raise Exception("tensor shape should be one")

    substring_len = substring.size(0)
    find_match = False
    for i in range(len(total_string) - substring_len + 1):
        window = total_string[i : i + substring_len]
        if torch.equal(window, substring):
            starting_index = i
            ending_index = i + substring_len - 1
            find_match = True
            break

    if find_match:
        logging.error(f"GOOD total string: {total_string}")
        logging.error(f"GOOD substring: {substring}")
        return starting_index, ending_index
    else:
        logging.error(f"total string: {total_string}")
        logging.error(f"substring: {substring}")
        return None, None


# Config
target_function = 'cos_sim' # cos_sim or euclidean
mode = 'white' # white or black box
device = "cuda"
num_iter = 20
# target_question = 'What is the longest river in the world?'
# target_answer = 'The Amazon River.'
embedding_model_type = "distilbert-base-uncased"
semantic_model_type = "cross-encoder/quora-distilroberta-base"
threshold_sem = 0.7
threshold_target_function = 0.7
target_json_file = "nq.json"
# crafted_json_file = mode + str(time.time()) + '.json'
crafted_json_file = 'test_euclidean.json'


# Load the models
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


def euclidean_distance(query1, query2):
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
    
    # euclidean distance
    return torch.norm(q1_sent_emb.squeeze() - q2_sent_emb.squeeze(), p=2)


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

    if target_function == 'cos_sim':
        # compute the cosine sim btw attacker and victim
        cos_sim = torch.nn.CosineSimilarity(dim=0)(
            attacker_sentence_embedding.squeeze(), victim_sentence_embedding.squeeze()
        )
        # backward!
        cos_sim.backward()
    elif target_function == 'euclidean':
        euclidean_distance = (-1) * torch.norm(
            attacker_sentence_embedding.squeeze() - victim_sentence_embedding.squeeze(), p=2
        )
        euclidean_distance.backward()
        

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


def emb_get_filtered_candidates(
    control_candidates, current_control, current_query, base, suffix_sem
):
    logging.debug("\n==========get_filtered_candidates===============")
    logging.debug(f"current_control: {current_control}")
    logging.debug(f"new_adv_suffix_toks: {control_candidates}")


    logging.debug(f"current_query: {current_query}")

    cands = []
    for i in range(control_candidates.shape[0]):
        decoded_str = embedding_tokenizer.decode(
            control_candidates[i], skip_special_tokens=True
        )
        new_query = base + decoded_str + suffix_sem
        if (
            decoded_str != current_control
            and len(
                embedding_tokenizer(decoded_str, add_special_tokens=False)["input_ids"]
            )
            == len(control_candidates[i])
        ):
            cands.append(decoded_str)

    if len(cands) == 0:
        cands.append(current_control)
        return cands
    
    # cands = cands + [cands[-1]] * (len(control_candidates) - len(cands))

    logging.debug("===========get_filtered_candidates===============\n")
    return cands


def emb_get_logits(
    suffix_semantic,
    control_start,
    control_end,
    test_controls,
    attacker_base,
    victim_query,
):
    logging.debug("\n==========get_logits===============")

    logging.debug(f"control_start:{control_start}, control_end:{control_end}")
    logging.debug(
        f"first of test controls: {test_controls[0]}, last of test controls: {test_controls[-1]}"
    )

    max_score = -500
    best_suffix = None
    for i in range(len(test_controls)):
        this_suffix = test_controls[i]
        new_attacker_query = attacker_base + this_suffix + suffix_semantic
        new_attacker_sen_emb = _sent_embed_from_text(new_attacker_query)
        victim_sentence_embedding = _sent_embed_from_text(victim_query)
        if target_function == 'cos_sim':
            score = torch.nn.CosineSimilarity(dim=0)(
                new_attacker_sen_emb, victim_sentence_embedding
            )
        elif target_function == 'euclidean':
            score = (-1) * torch.norm(
                new_attacker_sen_emb.squeeze() - victim_sentence_embedding.squeeze(), p=2
            )

        if score > max_score:
            max_score = score
            best_suffix = this_suffix

    logging.debug("==========get_logits===============\n")
    return max_score, best_suffix


def find_suffix_indices(attacker: str, suffix: str) -> list[int]:
    """Find the starting and ending indices of two identical suffix occurrences in the tokenized attacker string."""
    # Initialize model and tokenizer
    tokenizer = semantic_tokenizer

    # Tokenize attacker
    attacker_tokenized = tokenizer(
        attacker, return_tensors="pt", padding=True, return_offsets_mapping=True
    )
    attacker_ids = attacker_tokenized["input_ids"][0].tolist()

    # Tokenize suffix with a leading space to match attacker's context
    victim_tokenized = tokenizer(
        " " + suffix, return_tensors="pt", padding=True, return_offsets_mapping=True
    )
    # Strip special tokens (<s>, </s>) and trailing space
    victim_ids = victim_tokenized["input_ids"][0].tolist()[
        1:-2
    ]  # Core content without trailing space
    
    logging.error(f"attacker_ids: {attacker_ids}")
    logging.error(f"victim_ids: {victim_ids}")

    # Function to find all subsequence occurrences
    def find_all_subsequences(needle, haystack):
        n = len(needle)
        positions = []
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                positions.append(i)
        return positions

    # Find starting positions
    positions = find_all_subsequences(victim_ids, attacker_ids)

    # Validate that exactly two occurrences are found
    if len(positions) != 2:
        raise ValueError(
            f"Expected exactly 2 occurrences of suffix, found {len(positions)}"
        )

    # Calculate starting and ending indices
    suffix_length = len(victim_ids)
    start1 = positions[0]
    end1 = start1 + suffix_length - 1
    start2 = positions[1]
    end2 = start2 + suffix_length - 1

    return [start1, end1, start2, end2]


def craft_malicious(target_question, target):
    # Preprocess target question
    target_question = target_question.strip() + " "

    # Prepare attacker prompt
    victim_query = target_question
    attacker_query_base = target

    # Prepare suffix for embedding and semantic
    suffix_embedding = '!!!!!!!!!!!!!!!!!!!!!!'
    attacker_query = attacker_query_base + suffix_embedding
    
    # tokenize attack query by embedding tokenizer
    attacker_tokenized = get_input_tokenized_embedding(attacker_query)['input_ids'][0].tolist()
    suffix_tokenized = get_input_tokenized_embedding(suffix_embedding, special_tokens=False)['input_ids'][0].tolist()
    logging.debug(f"atacker: {attacker_tokenized}")
    logging.debug(f"suffix: {suffix_tokenized}")
    

    # decide the starting and ending index for adv suffix
    suffix_embedding_start = None
    suffix_embedding_end = None
    embedding_target_token_id = suffix_tokenized[0]
    for i in range(len(attacker_tokenized)):
        if attacker_tokenized[i] == embedding_target_token_id:
            if not suffix_embedding_start:
                suffix_embedding_start = i
            elif attacker_tokenized[i+1] != embedding_target_token_id:
                suffix_embedding_end = i
                break
            
    logging.debug(f"suffix_embedding_start: {suffix_embedding_start}")
    logging.debug(f"suffix_embedding_end: {suffix_embedding_end}")            
    

    # Gradient attack
    best_sim = 0
    best_attack = attacker_query
    for _ in range(num_iter):

        # Record best attacker query
        # if semantic_score(attacker_query, victim_query) >= threshold_sem:
        # TODO: we skip semantic evaluation for the time beding
        score = cosine_sim(attacker_query, victim_query) if target_function == 'cos_sim' else euclidean_distance(attacker_query, victim_query)
        if score >= threshold_target_function:
            if score >= best_sim:
                best_sim = score
                best_attack = attacker_query

        if None in [
            suffix_embedding_start,
            suffix_embedding_end,
        ]:
            logging.error("trouble analyzing suffix, two models interfere!")
            return

        # Update
        attacker_tokenized = get_input_tokenized_embedding(attacker_query)
        attacker_ids = attacker_tokenized["input_ids"]
        attacker_mask = attacker_tokenized["attention_mask"]

        embed_suffix_gradient = embed_token_gradient(
            attacker_ids,
            attacker_mask,
            suffix_embedding_start,
            suffix_embedding_end,
            victim_query,
        )
        
        
        with torch.no_grad():
            adv_control_tokens = attacker_ids[
                :, suffix_embedding_start : suffix_embedding_end + 1
            ]
            logging.debug(f"{adv_control_tokens}")

            new_adv_suffix_toks = sample_control(
                adv_control_tokens, embed_suffix_gradient, batch_size=512, topk=256
            )
            
        
            new_adv_suffix_text = emb_get_filtered_candidates(
                new_adv_suffix_toks,
                current_control=suffix_embedding,
                current_query=attacker_query,
                base=attacker_query_base,
                suffix_sem="",
            )
            logging.debug(f"{new_adv_suffix_text}")
            

            max_score, best_suffix = emb_get_logits(
                suffix_semantic="",
                control_start=suffix_embedding_start,
                control_end=suffix_embedding_end,
                test_controls=new_adv_suffix_text,
                attacker_base=attacker_query_base,
                victim_query=victim_query,
            )
            
            logging.info(f"the best suffix: {best_suffix}")

            suffix_embedding = best_suffix
            attacker_query = (
                attacker_query_base + suffix_embedding
            )
            logging.info(
                f"{target_function}: {max_score}, the attacker query: {repr(attacker_query)}"
            )
            # print(
            #     f"{target_function}: {max_score}, the attacker query: {repr(attacker_query)}"
            # )


    print(f"the best attack: {best_attack}")
    print(f"cos sim: {best_sim}")
    print(f"sem score: {semantic_score(best_attack, victim_query)}")
    return best_attack


if __name__ == "__main__":
    set_seed()
    set_logging()

    # load json
    # with open(target_json_file, "r") as f:
    #     target_data = json.load(f)

    # Craft for each target
    # for item in target_data.values():
    #     if 'adv_texts' in item:
    #         del item['adv_texts']

    #     target_question = item['question']
    #     target_answer = item['incorrect answer']
    #     if mode == 'white':
    #         attacker_query = craft_malicious_white_box(target_question, target_answer)
    #     else:
    #         attacker_query = craft_malicious_black_box(target_question, target_answer)
    #     item['attacker_query'] = attacker_query
        
    #     # Save the updated JSON
    #     with open(crafted_json_file, "w") as f:
    #         json.dump(target_data, f, indent=4)
    
    
    victim_target = {
    1: ("A cat sleeping on a bookshelf", "A rocket launching into space"),
    2: ("A chef chopping vegetables", "A dolphin jumping out of the water"),
    3: ("A robot vacuum cleaning the floor", "A mountain climber reaching the summit"),
    4: ("A child blowing soap bubbles", "A medieval knight holding a sword"),
    5: ("A jellyfish floating in the deep sea", "A musician playing a violin on stage"),
    6: ("A construction worker using a jackhammer", "A butterfly resting on a flower"),
    7: ("A golden retriever chasing a frisbee", "A train moving through a snowy landscape"),
    8: ("A scientist mixing chemicals in a lab", "A cowboy riding a horse in the desert"),
    9: ("A giant panda eating bamboo", "A satellite orbiting Earth"),
    10: ("A street artist painting graffiti", "A penguin sliding on ice"),
    11: ("A sumo wrestler preparing to fight", "A ballerina performing a pirouette"),
    12: ("A hot air balloon floating in the sky", "A shark swimming near a coral reef"),
    13: ("A firefighter rescuing a kitten", "A chess grandmaster making a move"),
    14: ("A magician pulling a rabbit out of a hat", "An astronaut walking on the moon"),
    15: ("A baker decorating a wedding cake", "A dragon breathing fire"),
    16: ("A bee collecting nectar from a flower", "A race car speeding on a track"),
    17: ("A man fishing by a quiet lake", "A parrot talking to its owner"),
    18: ("A baby giggling in a crib", "A samurai sharpening a katana"),
    19: ("A marathon runner crossing the finish line", "A whale diving deep into the ocean"),
    20: ("A grandma knitting a sweater", "A Formula 1 pit crew changing tires in seconds"),
    }



    # target_q = "a photo of a cat"  # victim
    # target_a = "a photo of a dog"  # target
    # attacker = craft_malicious_white_box(target_q, target_a)
    # print(attacker)
    
    result = {}
    for index, item in victim_target.items():
        entry = {}
        
        victim = item[0]
        target = item[1]
        attacker = craft_malicious(victim, target)
        
        entry['victim'] = victim
        entry['target'] = target
        entry['attacker'] = attacker
        entry['embedding_similarity'] = float(cosine_sim(attacker, victim))
        entry['semantic_score'] = float(semantic_score(attacker, victim))
        
        result[index] = entry
        
        
        # store into a file
        with open('result.json', "w") as file:
            json.dump(result, file, indent=4)  # indent=4 makes it human-readable
