from datasets import load_dataset
import json

ds = load_dataset("G:/pretrained_models/typed-decisions", "all", split="test")   # benchmark
tr = load_dataset("G:/pretrained_models/typed-decisions", "all", split="train")  # training data
row = ds[0]

state = json.loads(row["state"])
questions = json.loads(row["questions"])
gold = json.loads(row["gold"])


print(row)
# print(row["category__label"], row["category__confidence"])
# print(gold["urgency"]["probabilities"])   # distribution over rubric levels
