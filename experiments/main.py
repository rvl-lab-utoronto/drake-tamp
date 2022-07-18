from experiments.blocks_world.run import run_blocks_world
from experiments.two_arm_blocks_world.run import run_blocks_world as run_two_arm_blocks_world
from experiments.kitchen.run import run_kitchen
from experiments.kitchen_less_axioms.run import run_kitchen_less_axioms
from experiments.hanoi.run import run_hanoi
from experiments.basement_blocks_world.run import run_basement_blocks_world
import argparse
import json
import os
domains = {
    'kitchen': run_kitchen,
    'blocks_world':run_blocks_world,
    'two_arm_blocks_world': run_two_arm_blocks_world,
    'hanoi': run_hanoi,
    'kitchen_less_axioms': run_kitchen_less_axioms,
    'basement_blocks_world': run_basement_blocks_world
}

example = '{"model_path":"/home/agrobenj/drake-tamp/model_files/blocksworld_V2_adaptive/best.pt"}'
def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        choices=list(domains.keys()),
        default="kitchen",
        help = "The name of the domain to test"
    )
    parser.add_argument(
        "--domain-options",
        type=str,
        help = "Domain specific options in json kwarg format"
    )
    parser.add_argument(
        "--problem-file",
        type=str,
        required=False,
        help = "A path to the .yaml problem file"
    )
    parser.add_argument(
        "--oracle-options",
        type=str,
        help = f"Keyword arguments passed to the model in json format, like {example}"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="adaptive",
        choices=["informedV2", "adaptive"],
        help = "Which algorithm do you want to run the trial with?"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=[
            "normal", # nothing is saved to index.json, no oracle is used
            "save", # no oracle is used, stats_path is saved to index.json
            "oracle", 
            "complexityV3",
            "complexityandstructure",
            "complexitycollector",
            "oracleexpansion",
            "oraclemodel",
            "model",
            "cachingmodel",
            "multiheadmodel",
            "complexityoracle",
            "statsablation",
            "multiheadmodelperception",
            "multiheadmodelperception2",
            "ploiablation"
        ],
    )
    parser.add_argument(
        "--use-unique",
        action="store_true",
        help = "Make INFORMED use strictly unique results (refined) during planning. Warning, this will usually drasically slow things down"
    )
    parser.add_argument(
        "--should-save",
        action="store_true",
        help = "Force the oracle to save data to the save path"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=60,
        help="Maximum total runtime of the trial"
    )
    parser.add_argument(
        '--eager-mode',
        action="store_true",
        help = "Do you want to run INFORMED in eager-mode? (i.e every stream result popped of the queue has all it's children added to the I^*"
    )
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        required=False,
        help = "A path to (optionally) save a .profile file to (from CProfile)"
    )
    parser.add_argument(
        '--url',
        type=str, required=False,
        help="A meshcat url for viewing the problem"
    )
    parser.add_argument(
        '--logpath',
        type=str,
        default='logs',
        required=False,
        help="The directory to save the logs"
    )
    parser.add_argument(
        '--max_planner_time',
        type=float,
        default=10,
        required=False,
        help="The maximum time before FastDownward times out (per call)"
    )
    return parser

if __name__ == '__main__':
    args = make_argument_parser().parse_args()
    run_exp = domains[args.domain]
    domain_options = json.loads(args.domain_options) if args.domain_options else {}
    oracle_options = json.loads(args.oracle_options) if args.oracle_options else {}
    if args.problem_file:
        domain_options['problem_file'] = args.problem_file
    if args.logpath:
        if not os.path.isdir(args.logpath):
            os.mkdir(args.logpath)
    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable() 

    file = os.path.join(args.logpath, "run-params.txt")
    with open(file, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Domain options: {args.domain_options}\n")
        f.write(f"Use unique: {args.use_unique}\n")
        f.write(f"Max Time: {args.max_time}\n")
        f.write(f"Max Planner Time: {args.max_planner_time}\n")
        f.write(f"Logpath: {args.logpath}\n")
        f.write(f"Eager Mode: {args.eager_mode}\n")
        f.write(f"Domain: {args.domain}\n")
        f.write(f"Problem File: {args.problem_file}\n")

    run_exp(
        mode=args.mode,
        algorithm=args.algorithm,
        use_unique=args.use_unique,
        max_time=args.max_time,
        eager_mode=args.eager_mode,
        should_save=args.should_save,
        url=args.url if args.url else None,
        simulate=False,
        oracle_kwargs=oracle_options,
        path=args.logpath,
        max_planner_time = args.max_planner_time,
        **domain_options
    )
    if args.profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.print_stats()
        ps.dump_stats(args.profile)   