from make_problem import make_random_problem
import yaml
import os

if __name__ == "__main__":

    i = 0
    for num_cabbages in range(1, 4):
        for num_raddishes in range(0, 4):
            for num_glasses in range(0, 3):
                num_obj = num_cabbages + num_raddishes + num_glasses
                for num_goal in range(max(num_obj -3, 1), num_obj + 1):
                    yaml_data = make_random_problem(
                        num_cabbages = num_cabbages, num_raddishes = num_raddishes, num_glasses = num_glasses, prob_sink = 0
                    )
                    path = os.path.join("train", f"{num_cabbages}_{num_raddishes}_{num_glasses}_{num_goal}_{i}.yaml")
                    with open(path, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
                    print(f"Written {i+1}: {path}")
                    i +=1
    print(f"Num written: {i+1}")
    print(f"Max collection time: {(i+1)*90*2/60/60} hours")

    i  = 0

    for num_cabbages in range(1, 5):
        for num_raddishes in range(0, 4):
            for num_glasses in range(0, 3):
                num_obj = num_cabbages + num_raddishes + num_glasses
                for num_goal in range(max(num_obj -3, 1), num_obj + 1):
                    yaml_data = make_random_problem(
                        num_cabbages = num_cabbages, num_raddishes = num_raddishes, num_glasses = num_glasses, prob_sink = 0
                    )
                    path = os.path.join("test", f"{num_cabbages}_{num_raddishes}_{num_glasses}_{num_goal}_{i}.yaml")
                    with open(path, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
                    print(f"Written {i+1}: {path}")
                    i +=1
    print(f"Num written: {i+1}")
    print(f"Max collection time: {(i+1)*90*2/60/60} hours")
