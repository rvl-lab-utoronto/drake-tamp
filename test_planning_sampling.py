
#%%
from learning.cg_toy_simple import visualize
from integrate_planner import generate_scene, instantiate_planning_problem, externals
from lifted_search import ActionStreamSearch, repeated_a_star, Atom, try_a_star

scene = generate_scene([1, 2, 3, 4])
visualize(*scene)
#%%
def heuristic(state, goal):
    return len(goal - state.state)*10


#%%
objects, actions, initial_state = instantiate_planning_problem(scene)
goal = set()
goal.add(Atom('on', (objects['b0'].pddl, objects['r2'].pddl)))
goal.add(Atom('on', (objects['b1'].pddl, objects['r1'].pddl)))
goal.add(Atom('on', (objects['b2'].pddl, objects['r1'].pddl)))
goal.add(Atom('on', (objects['b3'].pddl, objects['r1'].pddl)))

search = ActionStreamSearch(initial_state, goal, externals, actions)
stats = {}
result = repeated_a_star(search, stats=stats, max_steps=10, heuristic=heuristic)
if result is not None:
    action_skeleton, object_mapping, _ = result
    actions_str = "\n".join([str(a) for a in action_skeleton])
    print(f"Action Skeleton:\n{actions_str}")
    print(f"\nObject mapping: {object_mapping}\n") 
# %%
import numpy as np
from copy import deepcopy
def visualize_plan(scene, objects, action_skeleton, object_mapping):
    visualize(*scene)
    world, grippers, regions, blocks = deepcopy(scene)
    objects_from_pddl = {v.pddl:v.value for o,v in objects.items()}
    for action in action_skeleton[1:]:
        if 'pick' in action.name:
            block = action.name.split(',')[1]
            # for otherblock in set(blocks) - {block}:
            #     print(otherblock, action.var_mapping)
            #     pose_pddl = action.var_mapping[f'?{otherblock}pose']
            #     pose = objects_from_pddl[pose_pddl] if pose_pddl in objects_from_pddl else object_mapping[pose_pddl]
            #     expected_pose = [blocks[otherblock]['x'], blocks[otherblock]['y']]
            #     assert np.allclose(pose - expected_pose), (pose, expected_pose)
            grippers['g1']['x'], grippers['g1']['y'] = object_mapping[action.var_mapping[f'?conf']].value
        elif 'place' in action.name:
            block = action.name.split(',')[1]
            print(block)
            grippers['g1']['x'], grippers['g1']['y'] = object_mapping[action.var_mapping[f'?conf']].value
            blocks[block]['x'], blocks[block]['y'] = object_mapping[action.var_mapping[f'?blockpose']].value
        visualize(world, grippers, regions, blocks)


# %%
visualize_plan(scene, objects, action_skeleton, object_mapping)
# %%
import time
start = time.time()
search = ActionStreamSearch(initial_state, goal, externals, actions)
result = repeated_a_star(search, stats=stats, max_steps=1, heuristic=heuristic)
duration = time.time() - start
print('Duration:', duration, 'seconds')
# %%
# Why 
from lifted_search import try_a_star
# cost = lambda state, op, child: 1 / (child.num_successes / child.num_attempts)
def cost(state, op, child, verbose=False):
    included = set()
    c = 0
    for obj in child.object_stream_map:
        stream_action = child.object_stream_map[obj]
        if stream_action is not None:
            if stream_action in included:
                continue
            cg_key = child.get_object_computation_graph_key(obj)
            if cg_key in stats:
                s = stats[cg_key]
                comp_cost = ((s['num_successes'] + 1) / (s['num_attempts'] + 1))**-1
                c += comp_cost

            else:
                comp_cost = 1
                c += comp_cost
            if verbose:
                print(stream_action, comp_cost)
            included.add(stream_action)
    return max(1, c)

new_search = ActionStreamSearch(initial_state, goal, externals, actions)
goal_state = try_a_star(new_search, cost, heuristic)
path = goal_state.get_path()
c = 0
for idx, i in enumerate(path):
    print(idx, i[1])
    a = cost(*i, verbose=True)
    print('action cost:', a)
    print('cum cost:', a+c)
    c += a
# %%
