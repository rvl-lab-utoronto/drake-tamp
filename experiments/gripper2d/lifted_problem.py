from lifted.a_star import ActionStreamSearch, repeated_a_star
import os
import sys

from pddlstream.algorithms.algorithm import parse_stream_pddl
from pddlstream.language.conversion import obj_from_value_expression
sys.path.insert(0, '/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
sys.path.insert(0, '/home/atharv/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
import pddl
from pddl.conditions import Atom, Conjunction, NegatedAtom
from pddl.effects import Effect
from pddl.actions import Action
from pddl.pddl_types import TypedObject
import pddl.conditions as conditions
from pddlstream.language.object import Object, OptimisticObject
from experiments.gripper2d.problem import ik, check_safe, grasp, placement, generate_scene
from pddlstream.language.generator import from_gen_fn, from_test


file_path, _ = os.path.split(os.path.realpath(__file__))

stream_pddl = open(f"{file_path}/stream.pddl", "r").read()

def scene_objects(world, grippers, regions, blocks):
    Object.reset()
    objects = {}
    for region_name in regions:
        objects[region_name] = Object(region_name)
    for gripper_name in grippers:
        objects[gripper_name] = Object(gripper_name)
        objects[f"{gripper_name}_pose"] = Object([grippers[gripper_name]['x'], grippers[gripper_name]['y']])
    for block_name in blocks:
        objects[block_name] = Object(block_name)
        objects[f"{block_name}_pose"] = Object([blocks[block_name]['x'], blocks[block_name]['y']])
    return objects

def instantiate_planning_problem(scene):
    world, grippers, regions, blocks = scene
    objects = scene_objects(*scene)
    actions = []
    for g in grippers:
        for r in regions:
            for bidx, b in enumerate(blocks):
                params = [
                        TypedObject(
                            name='?blockpose',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?conf',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?grasp',
                            type_name='object'
                        )
                ] + [TypedObject(name=f'?{other}pose', type_name='object') for other in blocks if other != b ]
                preconds = [
                            Atom('empty', (objects[g].pddl,)),
                            Atom('on', (objects[b].pddl, objects[r].pddl)),
                            Atom('atpose', (objects[b].pddl, '?blockpose')),
                            Atom('grasp', (objects[g].pddl,objects[b].pddl, '?grasp')),
                            Atom('ik', (objects[g].pddl, objects[b].pddl, '?conf', '?blockpose', '?grasp'))
                    ]
                for other in blocks:
                    if other != b:
                        preconds.append(Atom('atpose', (objects[other].pddl, f"?{other}pose")))
                        preconds.append(Atom('safe', (objects[g].pddl, '?conf', objects[other].pddl, f"?{other}pose")))
                effects = [
                    Atom('empty', (objects[g].pddl,)).negate(),
                    Atom('on', (objects[b].pddl, objects[r].pddl)).negate(),
                    Atom('atpose', (objects[b].pddl, '?blockpose')).negate(),
                    Atom('grasped', (objects[g].pddl, objects[b].pddl, '?grasp')),
                    Atom('holding', (objects[g].pddl, objects[b].pddl)),
                    Atom('atconf', (objects[g].pddl, '?conf')),
                ]
                actions.append(
                    Action(
                        name=f"pick({g},{b},{r})",
                        parameters=params,
                        num_external_parameters=len(params),
                        precondition=Conjunction(preconds),
                        effects=[Effect([], conditions.Truth(), atom) for atom in effects],
                        cost=1,
                    )
                )

                params = [
                        TypedObject(
                            name='?blockpose',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?conf',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?grasp',
                            type_name='object'
                        )
                ] + [TypedObject(name=f'?{other}pose', type_name='object') for other in blocks if other != b ]
                preconds = [
                            Atom('empty', (objects[g].pddl,)).negate(),
                            Atom('grasped', (objects[g].pddl, objects[b].pddl, '?grasp')),
                            Atom('holding', (objects[g].pddl, objects[b].pddl)),
                            Atom('placement', (objects[b].pddl, objects[r].pddl, '?blockpose')),
                            Atom('ik', (objects[g].pddl, objects[b].pddl, '?conf', '?blockpose', '?grasp'))
                    ]
                for other in blocks:
                    if other != b:
                        preconds.append(Atom('atpose', (objects[other].pddl, f"?{other}pose")))
                        preconds.append(Atom('safe', (objects[g].pddl, '?conf', objects[other].pddl, f"?{other}pose")))
                        preconds.append(Atom('safe-block', (objects[b].pddl, '?blockpose', objects[other].pddl, f"?{other}pose")))
                effects = [
                    Atom('empty', (objects[g].pddl,)),
                    Atom('grasped', (objects[g].pddl, objects[b].pddl, '?grasp')).negate(),
                    Atom('holding', (objects[g].pddl, objects[b].pddl)).negate(),
                    Atom('on', (objects[b].pddl, objects[r].pddl)),
                    Atom('atpose', (objects[b].pddl, '?blockpose')),
                    Atom('atconf', (objects[g].pddl, '?conf')),
                ]
                actions.append(
                    Action(
                        name=f"place({g},{b},{r})",
                        parameters=params,
                        num_external_parameters=len(params),
                        precondition=Conjunction(preconds),
                        effects=[Effect([], conditions.Truth(), atom) for atom in effects],
                        cost=1,
                    )
                )

    initial_state = set()
    for r in regions:
        initial_state.add(Atom('region', (objects[r].pddl, )))
    for g in grippers:
        initial_state.add(Atom('gripper', (objects[g].pddl, )))
        initial_state.add(Atom('empty', (objects[g].pddl,)))
        initial_state.add(Atom('atconf', (objects[g].pddl, objects[f"{g}_pose"].pddl)))
        initial_state.add(Atom('conf', (objects[f"{g}_pose"].pddl, )))
    for b in blocks:
        initial_state.add(Atom('block', (objects[b].pddl, )))
        initial_state.add(Atom('on', (objects[b].pddl, objects['r1'].pddl))) # TODO: put this into the scene def
        initial_state.add(Atom('atpose', (objects[b].pddl,  objects[f"{b}_pose"].pddl)))
        initial_state.add(Atom('blockpose', (objects[f"{b}_pose"].pddl, )))
        initial_state.add(Atom('placement', (objects[b].pddl, objects['r1'].pddl, objects[f"{b}_pose"].pddl)))
    
    return objects, actions, initial_state

def create_problem(scene, goal):
    objects, actions, initial_state = instantiate_planning_problem(scene)
    world, grippers, regions, blocks = scene
    stream_map = {
        "grasp": from_gen_fn(lambda g,b: grasp(grippers[g], blocks[b])),
        "ik": from_gen_fn(lambda g,_,bp,gr: ik(grippers[g], bp, gr, world)),
        "placement": from_gen_fn(lambda b,r: placement(blocks[b], regions[r])),
        "safe": from_test(lambda g,c,b,p: check_safe(grippers[g], c, blocks[b], p)),
        "safe-block": from_test(lambda b1,p1,b,p: check_safe(blocks[b1], p1, blocks[b], p)),
    }
    externals = parse_stream_pddl(
        stream_pddl,
        stream_map
    )
    if goal[0] != 'and':
        raise NotImplemented('Limited goal parsing. Expects a conjunction.')


    goal_set = set()
    for item in goal[1:]:
        pred = item[0]
        args = item[1:]
        goal_set.add(Atom(pred, tuple(objects[arg].pddl for arg in args)))
    
    return initial_state, goal_set, externals, actions

if __name__ == '__main__':
    import numpy as np
    import time
    np.random.seed(1)
    scene = generate_scene([1, 2, 3, 4])
    initial_state, goal, externals, actions = create_problem(scene, ('and', ('on', 'b0', 'r2'), ('on', 'b1', 'r1'), ('on', 'b2', 'r1'), ('on', 'b3', 'r1')))
    search = ActionStreamSearch(initial_state, goal, externals, actions)
    stats = {}

    profile = 'new_lifted.profile'
    if profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
    result = repeated_a_star(search, stats=stats, max_steps=10, heuristic=lambda s, g: len(g - s.state)*10)
    if result is not None:
        action_skeleton, object_mapping, _ = result
        actions_str = "\n".join([str(a) for a in action_skeleton])
        print(f"Action Skeleton:\n{actions_str}")
        print(f"\nObject mapping: {object_mapping}\n") 
    if profile:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.print_stats()
            ps.dump_stats(profile)   