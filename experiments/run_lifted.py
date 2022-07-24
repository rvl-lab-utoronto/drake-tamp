
# %%
from glob import glob
import json
import os
import yaml
import pickle

import getpass
USER = getpass.getuser()

import sys
sys.path.insert(
    0,
    f"/home/{USER}/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/",
)

from experiments.blocks_world.run_lifted import create_problem
from lifted.a_star import ActionStreamSearch, goalcount_heuristic, repeated_a_star, stream_cost, try_a_star, try_beam, try_policy_guided_beam, try_policy_guided
from learning.policy import make_policy, load_model, extract_labels
from pddlstream.language.object import Object
from experiments.blocks_world.pyper_utils import BlocksWorldPyperTranslator

#%%
if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", "-o", type=str, default=None)
    parser.add_argument("--problem_file", "-f", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--problem_type", "-p", type=str, default="random", choices=["random", "distractor", "clutter", "sorting", "stacking", "easy_distractors", "non_monotonic_v2"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--policy_path", type=str, default="policy.pt")
    parser.add_argument("--search_type", type=str, default="astar", choices=["beam", "astar"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--edge_eval_steps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--no_exclusion", action="store_true")
    parser.add_argument("--no_closed_list", action="store_true")
    parser.add_argument("--use_policy", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_label_path", type=str, default=None)
    parser.add_argument("--heuristic", type=str, default=None, choices=["goalcount", "hff", "hadd", "lama"])

    args = parser.parse_args()

    search_types = {
        "astar": try_a_star if not args.use_policy else try_policy_guided,
        "beam": try_beam if not args.use_policy else try_policy_guided_beam,
    }

    search_func = search_types[args.search_type]

    model = None

    if args.use_policy:
        model = load_model(args.policy_path)

    def run_problem(problem_file_path):
        with open(problem_file_path, 'r') as f:
            problem_file = yaml.safe_load(f)

        init, goal, externals, actions = create_problem(problem_file_path)

        if args.heuristic is not None:
            if args.heuristic == 'goalcount':
                h = goalcount_heuristic
            else:
                translator = BlocksWorldPyperTranslator()
                h = translator.create_heuristic(init, goal, args.heuristic)
        else:
            h = None

        search = ActionStreamSearch(init, goal, externals, actions)
        if search.test_goal(search.init):
            return {"solved": True, "expand_count": 0}

        stats = {}
        policy = make_policy(problem_file, search, model, stats) if args.use_policy else None

        r = repeated_a_star(
            search,
            search_func=search_func,
            beam_size=args.beam_size,
            stats=stats,
            policy_ts=policy,
            cost=stream_cost,
            max_steps=100,
            edge_eval_steps=args.edge_eval_steps,
            max_time=args.timeout,
            debug=args.debug,
            exclude_sampled_states_from_closed_list=not args.no_exclusion,
            use_closed=not args.no_closed_list,
            heuristic=h
        )
        if args.save_label_path:
            path_data = []
            for path in r.skeletons:
                path_data.append(extract_labels(path, r.stats))
            objects = ({
                o_pddl: dict(name=o_pddl,value=Object.from_name(o_pddl).value) 
                for o_pddl in search.init_objects
            })
        else:
            objects = None
            path_data = None

        return dict(
            name=os.path.basename(problem_file_path),
            planning_time=r.planning_time,
            solved=r.solution is not None,
            num_samples=r.num_samples,
            expanded=r.expand_count,
            evaluated=r.evaluate_count,
            skeletons=len(r.skeletons),
            plan_length=len(r.action_skeleton) if r.solution is not None else None,
            skeleton_lengths=[len(skeleton) for skeleton in r.skeletons],
            # training data things
            objects=objects,
            path_data=path_data,
        )
    if args.problem_file:
        problems = [args.problem_file]
    else:
        problems = sorted(glob(f"/home/{USER}/drake-tamp/experiments/blocks_world/data_generation/{args.problem_type}/{args.split}/*.yaml"))

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    def exclude_keys(dictionary, exclude=["objects", "path_data"]):
        return {k:v for k,v in dictionary.items() if k not in exclude}
    
    if args.output_path:
        with open(args.output_path, 'w') as f:
            f.write(json.dumps(vars(args)))
            f.write("\n")

    data = {}
    for problem_file_path in problems:
        print(f'Running {problem_file_path}')
        data[problem_file_path] = run_problem(problem_file_path)
        if args.output_path:
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(exclude_keys(data[problem_file_path])))
                f.write("\n")
        else:
            print(json.dumps(exclude_keys(data[problem_file_path])))
    if args.save_label_path:
        with open(args.save_label_path, 'wb') as f:
            pickle.dump(data, f)
    if args.profile:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.print_stats()
            ps.dump_stats(args.profile)   


# %%
