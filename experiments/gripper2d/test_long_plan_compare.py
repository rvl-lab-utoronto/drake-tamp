from experiments.gripper2d.test_long_plan import *
from experiments.gripper2d.lifted_problem import create_problem
from experiments.gripper2d.run import create_problem as create_problem_pddlstream, solve
from lifted.a_star import ActionStreamSearch, repeated_a_star
import time
import sys
import os
import pandas as pd
data = []
stdout = sys.stdout 
with open(os.devnull, 'w') as f:
    sys.stdout = f
    for i in range(1, 3):
        for j in range(3):
            initial_state, goal_set, externals, actions, objects = create_problem(scene, goal[:i + 1])
            search = ActionStreamSearch(initial_state, goal_set, externals, actions)
            stats = {}
            start = time.time()
            result = repeated_a_star(search, stats=stats, max_steps=10, heuristic=lambda s,g: 0)
            end = time.time()
            data.append(dict(alg='lifted', duration=end - start, num_goals=i, solved=result is not None and result[1] is not None))



        
            problem = create_problem_pddlstream(scene, goal[:i + 1])
            start = time.time()
            plan, _, _ = solve(
                problem,
                algorithm='adaptive',
                logpath="/tmp/",
                verbose=False
            )

            end = time.time()
            data.append(dict(alg='adaptive', duration=end - start, num_goals=i, solved=plan is not None))

sys.stdout = stdout
df = pd.DataFrame(data)
print(df.pivot_table(columns='alg', index='num_goals', values=['duration', 'solved']))

    