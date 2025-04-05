import json
from collections import defaultdict


file_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.3/alibaba/different_embedding/failed_ms_marco.json'
with open(file_path, 'r') as f:
    data = json.load(f)

dist = defaultdict(int)
for item in data:
    value = item['text-embedding-v1']
    range_start = (value // 0.05) * 0.05
    dist[range_start] += 1

for r in sorted(dist.keys()):
    print(f"{r:.2f} - {r+0.05:.2f}: {dist[r]}")