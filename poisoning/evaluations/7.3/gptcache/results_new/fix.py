import re
def quote_only_print(text):
    # Matches ONLY print: <text> if <text> is not already quoted
    return re.sub(
        r"(ONLY print\s*)['\"]?.+['\"]?",
        lambda m: m.group(1) + m.group(0).split("ONLY print")[1].strip()
        if re.match(r"^\s*['\"].+['\"]\s*$", m.group(0).split("ONLY print")[1])
        else m.group(1) + "'" + m.group(0).split("ONLY print")[1].strip().strip("'\"") + "'",
        text
    )


import json
# inputfile = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/gptcache/results_new/E73_gptcache_black_ms_marco.json'
inputfile = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/gptcache/results_new/E73_gptcache_white_ms_marco.json"
with open(inputfile, 'r') as file:
    data = json.load(file)
    
print(len(data))
for item in data:
    white = item['white']
    item['white'] = quote_only_print(white)


with open(inputfile, 'w') as file:
    json.dump(data, file, indent=4)
