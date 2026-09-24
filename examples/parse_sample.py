import json

with open("typed-decisions-sample.json",'r',encoding='utf-8') as f:
    data = json.load(f)

print(data)
print(type(data['state']),data['state'])