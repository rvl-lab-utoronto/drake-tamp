from experiments.blocks_world.run import run_blocks_world
from experiments.two_arm_blocks_world.run import run_blocks_world as run_two_arm_blocks_world
from experiments.kitchen.run import run_kitchen
import argparse
import json
domains = {
    'kitchen': run_kitchen,
    'blocks_world':run_blocks_world,
    'two_arms_blocks_world': run_two_arm_blocks_world
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
    
    return parser

if __name__ == '__main__':
    args = make_argument_parser().parse_args()
    run_exp = domains[args.domain]
    domain_options = json.loads(args.domain_options)
    run_exp(
        mode=args.mode,
        algorithm=args.algorithm,
        use_unique=args.use_unique,
        max_time=args.max_time,
        eager_mode=args.eager_mode,
        url=None,
        simulate=False,
        **domain_options
    )
