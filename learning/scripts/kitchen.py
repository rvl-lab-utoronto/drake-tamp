from learning.gnn.data import make_data_info, query_data, get_pddl_key
import numpy as np
import json

make_data_info(write = True)
blocks_world = get_pddl_key('kitchen')
data = query_data(blocks_world, [])
print(f"Num kitchen problems: {len(data)}")

num_valid = min(15, len(data) // 5)
np.random.shuffle(data)
valid = data[-num_valid:]
train = data[:-num_valid]
with open('/home/agrobenj/drake-tamp/learning/data/experiments/new_kitchen.json', 'w') as f:
    json.dump(dict(
        train=train,
        validation=valid,
        experiment_description="New kitchen training set"
    ), f, indent=4, sort_keys=True)
