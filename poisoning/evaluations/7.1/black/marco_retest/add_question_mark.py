import json


file_name = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/marco_retest/ignore_no_repeat_ms_marco.json"
with open(file_name, 'r') as f:
    data = json.load(f)

for item in data:
    adv = item['adv']
    parts = adv.split('  Ignore')
    if len(parts) == 2:
        item['adv'] = f"{parts[0]}?  Ignore{parts[1]}"

with open(file_name, 'w') as f:
    json.dump(data, f, indent=4)