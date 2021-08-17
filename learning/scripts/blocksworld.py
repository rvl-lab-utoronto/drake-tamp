
#%%
from learning.gnn.data import query_data, get_pddl_key
import numpy as np
import json
# %%
blocks_world = get_pddl_key('blocks_world')
# %%
data = query_data(blocks_world, [])
print(len(data))

# %%
num_valid = min(15, len(data) // 5)
np.random.shuffle(data)
valid = data[-num_valid:]
train = data[:-num_valid]
# %%
with open('learning/data/experiments/blocksworld_V2_adaptive.json', 'w') as f:
    json.dump(dict(
        train=train,
        validation=valid,
        experiment_description="Blocks world experiment with new domain and data coming from adaptive"
    ), f, indent=4, sort_keys=True)
