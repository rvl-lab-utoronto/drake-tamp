from make_problem import make_random_problem
import yaml
import os
import numpy as np

if __name__ == "__main__":

    i = 1
    max_goal = 3
    while i < 500:
        num_cabbages = np.random.randint(1, 3 + 1)
        num_raddishes = np.random.randint(0, 3 + 1)
        num_glasses = np.random.randint(0, 1 + 1)
        num_obj = num_cabbages + num_raddishes + num_glasses

        num_goal = np.random.randint(1, min(max_goal + 1, num_obj + 1))
        yaml_data = make_random_problem(
            num_cabbages = num_cabbages, num_raddishes = num_raddishes, num_glasses = num_glasses, prob_sink = 0, num_goal=num_goal
        )
        path = os.path.join("train_new", f"{num_cabbages}_{num_raddishes}_{num_glasses}_{num_goal}_{i}.yaml")
        with open(path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"Written {i+1}: {path}")
        i +=1
    print(f"Num written: {i+1}")
    print(f"Max collection time: {(i+1)*90*2/60/60} hours")

    i  = 1
    max_goal = 6
    while i < 100:
        num_cabbages = np.random.randint(3, 8 + 1)
        num_raddishes = np.random.randint(3, 8 + 1)
        num_glasses = np.random.randint(2, 4 + 1)
        num_obj = num_cabbages + num_raddishes + num_glasses
        num_goal = np.random.randint(1, min(max_goal + 1, num_obj + 1))
        yaml_data = make_random_problem(
            num_cabbages = num_cabbages, num_raddishes = num_raddishes, num_glasses = num_glasses, prob_sink = 0, num_goal=num_goal
        )
        path = os.path.join("test_new", f"{num_cabbages}_{num_raddishes}_{num_glasses}_{num_goal}_{i}.yaml")
        with open(path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"Written {i+1}: {path}")
        i +=1
    print(f"Num written: {i+1}")
    print(f"Max collection time: {(i+1)*90*2/60/60} hours")
