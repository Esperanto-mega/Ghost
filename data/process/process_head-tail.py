import json
from collections import Counter
import numpy as np

INTER_PATH = "../Instruments/Instruments.inter.json"
INDEX_PATH = "../Instruments/Instruments.RQK.index.json"
OUTPUT_HEAD = "../Instruments/Instruments.head.json"
OUTPUT_TAIL = "../Instruments/Instruments.tail.json"

with open(INTER_PATH, 'r') as f:
    inter = json.load(f)

with open(INDEX_PATH, 'r') as f:
    index = json.load(f)

all_raw_id = [int(i) for i in index.keys()]

train_set = {}

for k, v in inter.items():
    train_set[k] = v[:-2]

freq = Counter(item for lst in train_set.values() for item in lst)
hist = Counter(freq.values())

total = sum(hist.values())
ratio_80 = 0.8 * total

x_vals = np.arange(0, max(hist) + 1)
y_cum  = np.cumsum([hist.get(x, 0) for x in x_vals])

idx = np.searchsorted(y_cum, ratio_80, side='left')
x_80 = int(x_vals[idx])

hot_items = [item for item, cnt in freq.items() if cnt > x_80]
cold_items = [i for i in all_raw_id if i not in hot_items]

with open(OUTPUT_HEAD, 'w', encoding='utf-8') as f:
    json.dump(list(hot_items), f, ensure_ascii=False, indent=2)
with open(OUTPUT_TAIL, 'w', encoding='utf-8') as f:
    json.dump(list(cold_items), f, ensure_ascii=False, indent=2)
