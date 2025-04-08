import json
from util import *

threshold_sem = 0.8
threshold_target_function = 0.8

def craft_malicious_white_box_semantic_only(target_question, adv_existing, black_score, suffix_len=10):
    # Preprocess target question
    # target_question = target_question.strip() + " "
    target_question = target_question.strip()

    # Prepare attacker prompt
    victim_query = target_question
    attacker_query_base = adv_existing

    # Prepare suffix for embedding and semantic
    suffix_embedding = ''
    suffix_semantic = 'w' * suffix_len

    # attacker_query = attacker_query_base + suffix_semantic
    # attacker_query =  suffix_semantic + attacker_query_base
    attacker_query =  attacker_query_base + suffix_semantic
    
    
    suffix_semantic_start, suffix_semantic_end = find_semantic_suffix_indices(attacker_query, suffix_embedding, suffix_semantic, attacker_query_base)
    if None in [
        suffix_semantic_start,
        suffix_semantic_end,
    ]:
        print(suffix_semantic_start)
        print(suffix_semantic_end)
        logging.error("trouble analyzing suffix, two models interfere!")
        return


    # Gradient attack
    best_sim = 0
    best_score = 0
    best_attack = attacker_query
    for _ in range(num_iter):
        # Is it the right way to do that?
        sem_score = semantic_score(victim_query, attacker_query)
        cos_sim = cosine_sim(victim_query, attacker_query)
        white_score = sem_score + cos_sim
        if white_score > black_score and sem_score > 0.8 and white_score > best_score:
            best_attack = attacker_query
            best_score = white_score
            

        attacker_tokenized = get_input_tokenized_semantic(attacker_query)
        attacker_ids = attacker_tokenized["input_ids"]
        attacker_mask = attacker_tokenized["attention_mask"]

        sem_suffix_gradient = sem_token_gradients_penalty2(
            attacker_ids,
            attacker_mask,
            suffix_semantic_start,
            suffix_semantic_end,
            attacker_query,
            victim_query,
        )
        with torch.no_grad():
            adv_control_tokens = attacker_ids[
                :, suffix_semantic_start : suffix_semantic_end + 1
            ]
            
            # logging.error(f"attacker ids: {attacker_ids}")
            # logging.error(f"adv_control_tokens: {adv_control_tokens}")
            
            new_adv_suffix_toks = sample_control(
                adv_control_tokens, sem_suffix_gradient, batch_size=512, topk=256
            )
            
            # logging.error(new_adv_suffix_toks)

            new_adv_suffix_text = sem_get_filtered_candidates(
                new_adv_suffix_toks,
                current_control=suffix_semantic,
                current_query=attacker_query,
                base=attacker_query_base,
                suffix_embed=suffix_embedding,
            )
            
            # logging.error(f"new_adv_suffix_text: \n{new_adv_suffix_text}")

            max_score, best_suffix = sem_get_logits(
                embedding_suffix=suffix_embedding,
                control_start=suffix_semantic_start,
                control_end=suffix_semantic_end,
                test_controls=new_adv_suffix_text,
                attacker_base=attacker_query_base,
                victim_query=victim_query,
            )

            suffix_semantic = best_suffix
            attacker_query = (
                # attacker_query_base + suffix_embedding + suffix_semantic
                # suffix_semantic + attacker_query_base
                attacker_query_base + suffix_semantic
            )
            logging.info(
                f"Semantic score: {max_score}, the attacker query: {repr(attacker_query)},"
            )

    score = semantic_score(victim_query, best_attack)
    best_sim = cosine_sim(best_attack, victim_query)
    return best_attack, float(best_sim), float(score)



input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/whitebox/prompts_black/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/whitebox/prompts_white/'
datasets = [
    # "squad",
    # "MedQuad-MedicalQnADataset",
    # "ms_marco"
    "click"
]


if __name__ == "__main__":
    set_seed()
    set_logging()    
    
    # craft white box prompt: it is injection workable and it has higher sum score than black box
    for dataset_id in datasets:
        input_file = input_path + f"{dataset_id}.json"
        output_file = output_path + f"{dataset_id}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        for item in data:
            black = item['black']
            question = item['question']
            
            # black box score
            black_score = item['cos_sim_black'] + item['sem_score_black']
            
            # craft whtie
            adv_new, sim_new, score_new = craft_malicious_white_box_semantic_only(question, black, black_score)      
            white_score = sim_new + score_new
            print(adv_new)
            print(sim_new)
            print(score_new)
            
            if white_score > black_score and score_new > 0.8:
                item['white'] = adv_new
                item['cos_sim_white'] = sim_new
                item['sem_score_white'] = score_new
            else:
                item['white'] = black
                item['cos_sim_white'] = item['cos_sim_black']
                item['sem_score_white'] = item['sem_score_black']
                

            with open(output_file, "w") as file:
                json.dump(data, file, indent=4)