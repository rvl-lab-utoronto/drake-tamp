from experiments.blocks_world.run import run_blocks_world
from experiments.two_arm_blocks_world.run import run_blocks_world as run_two_arm_blocks_world
from experiments.kitchen.run import run_kitchen
import argparse
import json
domains = {
    'kitchen': run_kitchen,
    'blocks_world':run_blocks_world,
    'two_arm_blocks_world': run_two_arm_blocks_world
}
def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        choices=list(domains.keys()),
        default="kitchen"
    )
    parser.add_argument(
        "--domain-options",
        type=str
    )
    parser.add_argument(
        "--oracle-options",
        type=str
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="adaptive",
        choices=["informedV2", "adaptive"]
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=[
            "normal",
            "save",
            "oracle",
            "complexityV3",
            "complexitycollector",
            "oracleexpansion",
            "oraclemodel",
            "model",
            "cachingmodel",
            "complexityoracle"
        ]
    )
    parser.add_argument(
        "--use-unique",
        action="store_true"
    )
    parser.add_argument(
        "--should-save",
        action="store_true"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=60
    )
    parser.add_argument(
        '--eager-mode',
        action="store_true"
    )
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        '--url',
        type=str, required=False
    )
    
    return parser

if __name__ == '__main__':
    args = make_argument_parser().parse_args()
    run_exp = domains[args.domain]
    domain_options = json.loads(args.domain_options) if args.domain_options else {}
    oracle_options = json.loads(args.oracle_options) if args.oracle_options else {}
    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable() 
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
        **domain_options
    )
    if args.profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.print_stats()
        ps.dump_stats('exp.profile')   