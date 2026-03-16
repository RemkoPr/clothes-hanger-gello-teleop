from flask import json
import numpy as np


file = "/storage/rproesma/clothes-hanger/datasets/ICRA2026/clothes-hanger-v3-raw/meta/episodes.jsonl"

# parse file into list of dicts
episodes = []
with open(file, "r") as f:
    for line in f:
        episode = json.loads(line)
        episodes.append(episode)
        print(episode)

# print average length and standard deviation
lengths = [episode["length"] for episode in episodes]
print(f"Average episode length: {sum(lengths[:170]) / len(lengths[:170])}")
print(f"Standard deviation: {np.std(lengths[:170])}")
print(f"Total length: {sum(lengths)}")