from make_hanoi import make_hanoi_problem, PEGS
import yaml
import os
FILEPATH, _ = os.path.split(os.path.realpath(__file__))


if __name__ == "__main__":

    # train: 

    min_discs = 2
    max_discs = 9
    adaptive_max_discs = 6
    pegs = set(PEGS)

    for start_peg in pegs: 
        # 3
        for end_peg in pegs:
            if start_peg == end_peg:
                continue
        # 3 x 2 = 6
            for num_discs in range(min_discs, max_discs + 1):
                # 6 x 5 = 30
                if num_discs <= adaptive_max_discs:
                    yaml_data = make_hanoi_problem(num_discs, start_peg, end_peg)
                    path = os.path.join(FILEPATH, "train", f"{start_peg}_{end_peg}_{num_discs}.yaml")
                    print(f"Writing {path}")
                    with open(path, "w") as stream:
                        yaml.dump(yaml_data, stream, default_flow_style=False)
                path = os.path.join(FILEPATH, "test", f"{start_peg}_{end_peg}_{num_discs}.yaml")
                print(f"Writing {path}")
                with open(path, "w") as stream:
                    yaml.dump(yaml_data, stream, default_flow_style=False)
                