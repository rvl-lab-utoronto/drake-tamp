
#%%
from learning.gnn.data import query_data, get_pddl_key, DifficultClasses
import numpy as np
import json
# %%
kitchen_key = get_pddl_key('kitchen')

# %%
easy = query_data(kitchen_key, DifficultClasses.easy)
medium = query_data(kitchen_key, DifficultClasses.medium)
hard = query_data(kitchen_key, DifficultClasses.hard)
vhard = query_data(kitchen_key, DifficultClasses.very_hard)

# %%
np.random.shuffle(easy)
np.random.shuffle(medium)
np.random.shuffle(hard)
np.random.shuffle(vhard)
num_valid_easy, num_valid_medium, num_valid_hard, num_valid_vhard = 10, 5, 5, 5
valid = easy[-num_valid_easy:]
valid += medium[-num_valid_medium:]
valid += hard[-num_valid_hard:]
valid += vhard[-num_valid_vhard:]

train = easy[:-num_valid_easy]
train += medium[:-num_valid_medium]
train += hard[:-num_valid_hard]
train += vhard[:-num_valid_vhard]
np.random.shuffle(train)
np.random.shuffle(valid)

# %%
with open('learning/data/experiments/kitchen_diffclasses.json', 'w') as f:
    json.dump(dict(
        train=train,
        validation=valid,
        experiment_description="Prelim testing with kitchen dataset using time-based difficulty classes. Validation has a slightly higher proportion of hard runs. This time, we have what we need for a problem graph model which encodes scene graph and goal."
    ), f, indent=4, sort_keys=True)
