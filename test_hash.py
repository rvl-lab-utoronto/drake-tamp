#%%
from learning.policy import *

def get_plan(action_names, search):
    state = search.init
    path = []
    while action_names:

        next_action_name = action_names.pop(0)
        for (action, next_state) in search.successors(state):
            if next_action_name in action.name:
                break
        else:
            raise ValueError(f'Could not find a successor with {next_action_name}')
        path.append((state, action, next_state))
        state = next_state
    return path

problem_file_path = 'experiments/blocks_world/data_generation/random/test/2_6_2_41.yaml'
init, goal, externals, actions = create_problem(problem_file_path)
search = ActionStreamSearch(init, goal, externals, actions)

path = get_plan([
    'pick(panda,blocker0,blue_table)',
    'place(panda,blocker0,blue_table)'
], search)
final = path[-1][-1]

print(search.init.id_key)
print()
print(final.id_key)

assert hash(search.init) != hash(final)
print('Success')
