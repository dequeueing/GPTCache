import json
import re
def quote_only_print(text):
    # Matches ONLY print: <text> if <text> is not already quoted
    return re.sub(
        r"(ONLY print:\s*)['\"]?.+['\"]?",
        lambda m: m.group(1) + m.group(0).split("ONLY print:")[1].strip()
        if re.match(r"^\s*['\"].+['\"]\s*$", m.group(0).split("ONLY print:")[1])
        else m.group(1) + "'" + m.group(0).split("ONLY print:")[1].strip().strip("'\"") + "'",
        text
    )


file_name = 'trivia.json'
with open(file_name, 'r') as file:
    data = json.load(file)
    
for item in data:
    question = item['question']
    white = item['white']
    white_new = quote_only_print(white)
    # print(white_new)
    item['white'] = white_new

with open(file_name, 'w') as file:
    json.dump(data, file, indent=4)
