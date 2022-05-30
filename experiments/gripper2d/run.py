#%%
from experiments.gripper2d.problem import ik, check_safe, grasp, placement, generate_scene
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.language.stream import StreamInfo
from pddlstream.algorithms.meta import solve
import os
file_path, _ = os.path.split(os.path.realpath(__file__))

domain_pddl = open(f"{file_path}/domain.pddl", "r").read()
stream_pddl = open(f"{file_path}/stream.pddl", "r").read()

def create_problem(scene, goal):
    world, grippers, regions, blocks = scene
    stream_map = {
        "grasp": from_gen_fn(lambda g,b: grasp(grippers[g], blocks[b])),
        "ik": from_gen_fn(lambda g,_,bp,gr: ik(grippers[g], bp, gr, world)),
        "placement": from_gen_fn(lambda b,r: placement(blocks[b], regions[r])),
        "safe": from_test(lambda g,c,b,p: check_safe(grippers[g], c, blocks[b], p)),
        "safe-block": from_test(lambda b1,p1,b,p: check_safe(blocks[b1], p1, blocks[b], p)),
    }
    init = set()
    for r in regions:
        init.add(('region', r))
    for g in grippers:
        p = tuple([grippers[g]['x'], grippers[g]['y']])
        init.add(('gripper', g))
        init.add(('empty', g))
        init.add(('atconf', g, p))
        init.add(('conf', p))
    for b in blocks:
        p = tuple([blocks[b]['x'], blocks[b]['y']])
        init.add(('block', b))
        init.add(('on', b, blocks[b]["on"]))
        init.add(('blockpose', p))
        init.add(('atpose', b,  p))
    
    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)
if __name__ == '__main__':
    scene = generate_scene([1, 2, 3, 4])
    goal = ('and', 
        ('on', 'b0', 'r2'),
        ('on', 'b1', 'r1'),
        ('on', 'b2', 'r1'),
        ('on', 'b3', 'r1')
    )
    problem = create_problem(scene, goal)

    solution = solve(
            problem,
            algorithm='adaptive',
            # use_unique=True,
            max_time=60,
            search_sample_ratio=1,
            max_planner_time = 30,
            logpath="/tmp/",
            stream_info={
                "grasp": StreamInfo(use_unique=True),   
                "ik": StreamInfo(use_unique=True),  
                "placement": StreamInfo(use_unique=True),   
                "safe": StreamInfo(use_unique=True),    
                "safe-block": StreamInfo(use_unique=True),  
            },
            verbose=False
        )
    print_solution(solution)

