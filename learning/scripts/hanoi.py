#%%
from learning.gnn.data import get_base_datapath, make_data_info, query_data, get_pddl_key
import numpy as np
import os
import json
import pickle
import glob


#%%
make_data_info(write = True)

#%%

hanoi = get_pddl_key('hanoi')
data = query_data(hanoi, [])

pos = 0
neg = 0
for d in data:
    dir = os.path.join(get_base_datapath(), os.path.splitext(d)[0])
    for label_path in glob.glob(os.path.join(dir, "*")):
        with open(label_path, "rb") as f:
            label = pickle.load(f)
            if label.label:
                pos+=1
            else:
                neg+=1
tot = pos + neg
print("Proportion of positive labels for hanoi:", float(pos)/tot)

#%%

#make_data_info(write = True)
kitchen = get_pddl_key('kitchen')
data = query_data(kitchen, [])

pos = 0
neg = 0
for d in data:
    dir = os.path.join(get_base_datapath(), os.path.splitext(d)[0])
    for label_path in glob.glob(os.path.join(dir, "*")):
        with open(label_path, "rb") as f:
            print(label_path)
            label = pickle.load(f)
            if label.label:
                pos+=1
            else:
                neg+=1
tot = pos + neg
print("Proportion of positive labels for kitchen:", float(pos)/tot)

#%%
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