import json
import torch
from util import *
from typing import *
import matplotlib.pyplot as plt

from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)



def check_inject_success(attacker_response, target_answer):
    attacker_response = attacker_response.rstrip('.')
    target_answer = target_answer.rstrip('.')
    return attacker_response == target_answer


input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/prompts_flick'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/prompts_flick'
datasets = {
    # "squad": "squad_targeted.json",
    # "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    # "ms_marco": "ms_marco_targeted.json",
    "flick": "flick"
}
# prompt_injection_patterns = [
#     'dont_answer_PI_', 
#     # 'ignore_PI_'
# ]

prompt_injection_patterns = [
    # 'dont_answer_no_repeat_', 
    # 'dont_answer_tripple_repeat_', 
    # 'ignore_PI_',
    'ignore_no_repeat_',
    # 'ignore_tripple_repeat_',
]

if __name__ == '__main__':
    # stat_file = output_path + f"E71_black_box_final_gptcache_summary.json"
    # with open(stat_file, 'r') as f:
    #     stat = json.load(f)

    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            
            input_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/prompts_flick/flick.json"
            output_file = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/prompts_flick/flick.json"
            
            # load data
            with open(input_file, 'r') as f:
                data = json.load(f)
                 
            total = len(data)
            cos_sim = 0
            sem_score = 0
            euc_dist = 0
            for item in data:
                injection_success = False
                attack_success = False
                similar_enough = False
                
                question = item['prompt']
                adv = item['white']
                # target_answer = item['wrong']
                
                cos_sim += cosine_sim(adv, question).item()
                sem_score += semantic_score(question, adv).item()
                euc_dist += euclidean_distance(adv, question).item()
                
                print(semantic_score(question, adv).item(), ',')
                
                
                # item['attacker_response'] = attacker_response
                # item['injection_success'] = injection_success
                # item['cos_sim'] = float(cos_sim)
                # item['sem_score'] = float(sem_score)
                # item['euc_dist'] = float(euc_dist)


                # with open(output_file, 'w') as file:
                #     json.dump(data, file, indent=4)
                
            print(f"black box result:")
            print(f"average cosime similarity: {cos_sim/total}")
            print(f"average semantic score: {sem_score/total}")
            print(f"average euclidean distance: {euc_dist/total}")
            
            
    data = [
        0.9536388516426086 ,
        0.9948493838310242 ,
        0.9644471406936646 ,
        0.9758715033531189 ,
        0.9832919239997864 ,
        0.8965404629707336 ,
        0.977020800113678 ,
        0.9106553196907043 ,
        0.9747739434242249 ,
        0.9505362510681152 ,
        0.9829645752906799 ,
        0.9952261447906494 ,
        0.9797366261482239 ,
        0.9694851636886597 ,
        0.9973082542419434 ,
        0.961955189704895 ,
        0.9699528217315674 ,
        0.9703034162521362 ,
        0.9777922630310059 ,
        0.9802055358886719 ,
        0.9625375270843506 ,
        0.9847404360771179 ,
        0.9842026233673096 ,
        0.9394842386245728 ,
        0.9891923069953918 ,
        0.8999364376068115 ,
        0.9910983443260193 ,
        0.39644694328308105 ,
        0.9783801436424255 ,
        0.9311304092407227 ,
        0.9562623500823975 ,
        0.9951080679893494 ,
        0.9828307628631592 ,
        0.9884032607078552 ,
        0.9697431325912476 ,
        0.866340696811676 ,
        0.8494046330451965 ,
        0.9949983358383179 ,
        0.9527470469474792 ,
        0.9386799335479736 ,
        0.9687542915344238 ,
        0.9932218790054321 ,
        0.9718786478042603 ,
        0.9738644361495972 ,
        0.980767011642456 ,
        0.9940345883369446 ,
        0.9598903059959412 ,
        0.992099404335022 ,
        0.9765457510948181 ,
        0.9960237741470337 ,
        0.9802373051643372 ,
        0.935567319393158 ,
        0.9712603092193604 ,
        0.9566847681999207 ,
        0.9361650347709656 ,
        0.03721987456083298 ,
        0.990844190120697 ,
        0.18812955915927887 ,
        0.9806232452392578 ,
        0.9628047943115234 ,
        0.21174275875091553 ,
        0.9630257487297058 ,
        0.7906869053840637 ,
        0.9766606688499451 ,
        0.9740124344825745 ,
        0.9770320057868958 ,
        0.6377041339874268 ,
        0.9412857890129089 ,
        0.9720484614372253 ,
        0.9646138548851013 ,
        0.9656592011451721 ,
        0.9786054491996765 ,
        0.969528317451477 ,
        0.9453326463699341 ,
        0.9635968804359436 ,
        0.9607277512550354 ,
        0.9564833641052246 ,
        0.967144250869751 ,
        0.9706947207450867 ,
        0.98310387134552 ,
        0.9824296236038208 ,
        0.9750654697418213 ,
        0.9710412621498108 ,
        0.8932647109031677 ,
        0.9920724034309387 ,
        0.9825873970985413 ,
        0.9655104875564575 ,
        0.9876624345779419 ,
        0.8651295304298401 ,
        0.9741746187210083 ,
        0.9634993672370911 ,
        0.9788697361946106 ,
        0.9979488253593445 ,
        0.9527495503425598 ,
        0.9678608775138855 ,
        0.9677020907402039 ,
        0.9627240896224976 ,
        0.9913777112960815 ,
        0.8335086107254028 ,
        0.9731560349464417 ,
    ]        

    plt.hist(data, bins=10, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Data')
    plt.savefig('histogram_of_data.png')


