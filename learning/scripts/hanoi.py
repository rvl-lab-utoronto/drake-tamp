from learning.gnn.data import make_data_info, query_data, get_pddl_key
import numpy as np
import os
import json

make_data_info(write = True)
hanoi = get_pddl_key('hanoi')
data = query_data(hanoi, [])
print(len(data))
num_valid = min(15, len(data) // 5)
np.random.shuffle(data)
valid = data[-num_valid:]
print("Num valid:", num_valid)
train = data[:-num_valid]
print("Num train", len(train))
with open(os.path.expanduser('~/drake-tamp/learning/data/experiments/hanoi.json'), 'w') as f:
    json.dump(dict(
        train=train,
        validation=valid,
        experiment_description="First trial of hanoi"
    ), f, indent=4, sort_keys=True)