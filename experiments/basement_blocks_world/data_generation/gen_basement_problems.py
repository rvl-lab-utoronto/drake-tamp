from make_problem import make_random_problem
import os
import yaml

if __name__ == "__main__":

    i = 0
    for num_blocks in range(2,8):
        for _ in range(2):
            max_stack = min(num_blocks, 6)
            for max_start_stack in range(1,max_stack +1):
                for max_goal_stack in range(2,max_stack +1):
                    yaml_data = make_random_problem(num_blocks, max_start_stack=max_start_stack, max_goal_stack=max_goal_stack)
                    path = os.path.join("train", f"{num_blocks}_{max_start_stack}_{max_goal_stack}_{i}.yaml")
                    with open(path, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
                    print(f"Written {i+1}: {path}")
                    i+=1
    
    i = 0
    for num_blocks in range(2,8):
        for _ in range(2):
            max_stack = min(num_blocks, 6)
            for max_start_stack in range(1,max_stack+1):
                for max_goal_stack in range(2,max_stack+1):
                    yaml_data = make_random_problem(num_blocks, max_start_stack=max_start_stack, max_goal_stack=max_goal_stack)
                    path = os.path.join("test", f"{num_blocks}_{max_start_stack}_{max_goal_stack}_{i}.yaml")
                    with open(path, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
                    print(f"Written {i+1}: {path}")
                    i+=1