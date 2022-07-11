
# %%
from glob import glob
import json
import os
import yaml

import getpass
USER = getpass.getuser()

import sys
sys.path.insert(
    0,
    f"/home/{USER}/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/",
)

from experiments.blocks_world.run_lifted import create_problem
from lifted.a_star import ActionStreamSearch, repeated_a_star, stream_cost, try_policy_guided, try_a_star
from learning.policy import make_policy, load_model


#%%
if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", "-o", type=str, default=None)
    parser.add_argument("--problem_file", "-f", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--problem_type", "-p", type=str, default="random", choices=["random", "distractor", "clutter", "sorting", "stacking", "easy_distractors"])
    parser.add_argument("--use_policy", action="store_true")
    args = parser.parse_args()

    if args.use_policy:
        search_func = try_policy_guided
    else:
        search_func = try_a_star

    output_path = args.output_path if args.output_path is not None else f"lifted{'_policy' if args.use_policy else ''}_{args.problem_type}_astar_{time.time()}_output.json"

    model = None

    if args.use_policy:
        model = load_model()

    def run_problem(problem_file_path, data):
        with open(problem_file_path, 'r') as f:
            problem_file = yaml.safe_load(f)

        init, goal, externals, actions = create_problem(problem_file_path)

        search = ActionStreamSearch(init, goal, externals, actions)
        if search.test_goal(search.init):
            data[problem_file_path] = {"solved": True, "expand_count": 0}
            return

        stats = {}
        policy = make_policy(problem_file, search, model, stats) if args.use_policy else None
        m_attempts = lambda a: 10**(1 + a // 10)
        def stream_cost_fn(s, o, c, stats=stats, verbose=False):
            fa = stream_cost(s, o, c, given_stats=stats)
            fa = (1 + fa["num_successes"] * m_attempts(fa["num_attempts"])) / (1 +  fa["num_attempts"]*m_attempts(fa["num_attempts"]))
            return 1/fa

        r = repeated_a_star(search, search_func=search_func, stats=stats, policy_ts=policy, cost=stream_cost_fn, max_steps=100, edge_eval_steps=10, max_time=120, debug=args.problem_file is not None)


        data[problem_file_path] = dict(
            name=os.path.basename(problem_file_path),
            planning_time=r.planning_time,
            solved=r.solution is not None,
            # goal=goal,
            # objects=objects,
            # path_data=path_data,
            num_samples=r.num_samples,
            expanded=r.expand_count,
            evaluated=r.evaluate_count,
            skeletons=len(r.skeletons)
        )
    if args.problem_file:
        problems = [args.problem_file]
    else:
        problems = sorted(glob(f"/home/{USER}/drake-tamp/experiments/blocks_world/data_generation/{args.problem_type}/test/*.yaml"))

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
    data = {}
    for problem_file_path in problems:
        run_problem(problem_file_path, data)

        with open(output_path, 'a') as f:
            f.write(json.dumps(data[problem_file_path]))
            f.write("\n")
    if args.profile:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.print_stats()
            ps.dump_stats(args.profile)   

