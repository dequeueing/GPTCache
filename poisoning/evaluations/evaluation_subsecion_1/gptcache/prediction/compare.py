import json

if __name__ == '__main__':
    origin = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/prediction/prediction_MedQuad-MedicalQnADataset.json'
    new = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evaluation_subsecion_1/gptcache/prediction/new_prediction_MedQuad-MedicalQnADataset.json'
    
    with open(origin, 'r') as f:
        origin = json.load(f)
    with open(new, 'r') as f:
        new = json.load(f)
        
    good = 0
    new_semantic_ok = 0
    old_semantic_ok = 0
    for i in range(len(origin)):
        origin_item = origin[i]
        new_item = new[i]

        origin_dist = origin_item['euclidean']
        new_dist = new_item['euclidean']
        if new_dist < origin_dist:
            good += 1
        
        if new_item['semantic'] > 0.8:
            new_semantic_ok += 1
        else:
            print(new_item)
        if origin_item['semantic'] > 0.8:
            old_semantic_ok += 1
            
    print(f"improve in euclidean distance: {good}")
    print(f"semantic ok: new: {new_semantic_ok}, old: {old_semantic_ok}")

