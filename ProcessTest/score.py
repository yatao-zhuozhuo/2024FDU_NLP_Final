import json

with open('scores_oral_MATH.jsonl', 'r', encoding='utf-8') as f:
    content = [json.loads(line) for line in f]

count = 0
score = 0

for item in content:
    count += 1
    score += item['score']

print("Average Score: ", score/count)