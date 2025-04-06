import json

files = [
    'blackbox.json',
    'whitebox.json',
]


for file_name in files:
    with open(file_name, 'r') as file:
        data = json.load(file)
        
    for index, item in enumerate(data):
        item['id'] = index
    
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
