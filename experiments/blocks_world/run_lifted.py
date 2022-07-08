#%%

from experiments.blocks_world.run import *
from pddlstream.algorithms.algorithm import parse_problem

from pddlstream.language.conversion import fact_from_evaluation
sys.path.insert(0, '/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
sys.path.insert(0, '/home/atharv/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
import pddl
from pddl.conditions import Atom, Conjunction, NegatedAtom
from pddl.effects import Effect
from pddl.actions import Action
from pddl.pddl_types import TypedObject
import pddl.conditions as conditions
from pddlstream.language.object import Object

def create_problem(problem_file):
    Object.reset()
    (
        sim,
        station_dict,
        traj_directors,
        meshcat_vis,
        prob_info,
    ) = make_and_init_simulation(None, problem_file)
    problem, model_poses = construct_problem_from_sim(sim, station_dict, prob_info)
    evaluations, goal_exp, domain, externals = parse_problem(problem)
    actions = instantiate_actions(prob_info)
    init = set()
    for evaluation in evaluations:
        x = fact_from_evaluation(evaluation)
        init.add(Atom(x[0], [o.pddl for o in x[1:]]))

    goal = set()
    assert goal_exp[0] == 'and'
    for x in goal_exp[1:]:
        goal.add(Atom(x[0], [o.pddl for o in x[1:]]))

    return init, goal, externals, actions

def instantiate_actions(prob_info):
    """Define partially grounded actions like pick(arm, block0, table1)
    Returns a list of actions.
    """
    actions = []
    for arm in prob_info.arms:
        arm_pddl = Object.from_value(arm).pddl
        for surface in prob_info.surfaces:
            table = (surface, prob_info.surfaces[surface][0])
            surface_pddl = Object.from_value(table).pddl
            for object in prob_info.objects:
                object_pddl = Object.from_value(object).pddl
                other_objects = [o for o in prob_info.objects if o != object]

                # PICK #
                params = [
                        TypedObject(
                            name='?X_WB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?X_HB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?pre_q',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?q',
                            type_name='object'
                        )
                ] + [TypedObject(name=f'?X_W{other}', type_name='object') for other in other_objects]
                preconds = [
                        Atom('clear', (object_pddl,)),
                        Atom('empty', (arm_pddl,)),
                        Atom('on-table', (object_pddl, surface_pddl)),
                        Atom('atworldpose', (object_pddl, '?X_WB')),
                        # Atom('atconf', (arm_pddl, '?pre_q')),
                        Atom('ik', (arm_pddl, object_pddl,'?X_WB', '?X_HB', '?pre_q', '?q'))
                ]
                for other in other_objects:
                    preconds.append(Atom('atworldpose', (Object.from_value(other).pddl, f"?X_W{other}")))
                    preconds.append(Atom('colfree-block', (arm_pddl, '?q', Object.from_value(other).pddl, f"?X_W{other}")))
    
                effects = [
                    Atom('empty', (arm_pddl,)).negate(),
                    Atom('on-table', (object_pddl, surface_pddl)).negate(),
                    Atom('atworldpose', (object_pddl, '?X_WB')).negate(),
                    Atom('athandpose', (arm_pddl, object_pddl, '?X_HB')),
                    Atom('grasped', (arm_pddl, object_pddl)),
                ]
                actions.append(
                    Action(
                        name=f"pick({arm},{object},{surface})",
                        parameters=params,
                        num_external_parameters=len(params),
                        precondition=Conjunction(preconds),
                        effects=[Effect([], conditions.Truth(), atom) for atom in effects],
                        cost=1,
                    )
                )

                # PLACE #
                params = [
                        TypedObject(
                            name='?X_WB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?X_HB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?pre_q',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?q',
                            type_name='object'
                        )
                ] + [TypedObject(name=f'?X_W{other}', type_name='object') for other in other_objects]
                preconds = [
                        Atom('ik', (arm_pddl, object_pddl,'?X_WB', '?X_HB', '?pre_q', '?q')),
                        Atom('athandpose', (arm_pddl, object_pddl, '?X_HB')),
                        Atom('grasped', (arm_pddl, object_pddl)),
                        # Atom('atconf', (arm_pddl, '?pre_q')),
                        Atom('table-support', (object_pddl, '?X_WB', surface_pddl)),
                ]
                for other in other_objects:
                    preconds.append(Atom('atworldpose', (Object.from_value(other).pddl, f"?X_W{other}")))
                    preconds.append(Atom('colfree-block', (arm_pddl, '?q', Object.from_value(other).pddl, f"?X_W{other}")))
    
                effects = [
                    Atom('empty', (arm_pddl,)),
                    Atom('on-table', (object_pddl, surface_pddl)),
                    Atom('atworldpose', (object_pddl, '?X_WB')),
                    Atom('athandpose', (arm_pddl, object_pddl, '?X_HB')).negate(),
                    Atom('grasped', (arm_pddl, object_pddl)).negate(),
                ]
                actions.append(
                    Action(
                        name=f"place({arm},{object},{surface})",
                        parameters=params,
                        num_external_parameters=len(params),
                        precondition=Conjunction(preconds),
                        effects=[Effect([], conditions.Truth(), atom) for atom in effects],
                        cost=1,
                    )
                )
        for object in prob_info.objects:
            object_pddl = Object.from_value(object).pddl
            other_objects = [o for o in prob_info.objects if o != object]
            for other_object in other_objects:
                other_object_pddl = Object.from_value(other_object).pddl
                # UNSTACK #
                params = [
                        TypedObject(
                            name='?X_WB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?X_HB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?pre_q',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?q',
                            type_name='object'
                        )
                ] + [TypedObject(name=f'?X_W{other}', type_name='object') for other in other_objects]
                preconds = [
                        Atom('clear', (object_pddl,)),
                        Atom('empty', (arm_pddl,)),
                        Atom('on-block', (object_pddl, other_object_pddl)),
                        Atom('atworldpose', (object_pddl, '?X_WB')),
                        # Atom('atconf', (arm_pddl, '?pre_q')),
                        Atom('ik', (arm_pddl, object_pddl,'?X_WB', '?X_HB', '?pre_q', '?q'))
                ]
                for other in other_objects:
                    preconds.append(Atom('atworldpose', (Object.from_value(other).pddl, f"?X_W{other}")))
                    preconds.append(Atom('colfree-block', (arm_pddl, '?q', Object.from_value(other).pddl, f"?X_W{other}")))
    
                effects = [
                    Atom('empty', (arm_pddl,)).negate(),
                    Atom('on-block', (object_pddl, other_object_pddl)).negate(),
                    Atom('atworldpose', (object_pddl, '?X_WB')).negate(),
                    Atom('athandpose', (arm_pddl, object_pddl, '?X_HB')),
                    Atom('clear', (other_object_pddl,)),
                    Atom('grasped', (arm_pddl, object_pddl)),

                ]
                actions.append(
                    Action(
                        name=f"unstack({arm},{object},{other_object})",
                        parameters=params,
                        num_external_parameters=len(params),
                        precondition=Conjunction(preconds),
                        effects=[Effect([], conditions.Truth(), atom) for atom in effects],
                        cost=1,
                    )
                )

                # STACK #
                params = [
                        TypedObject(
                            name='?X_WB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?X_HB',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?pre_q',
                            type_name='object'
                        ),
                        TypedObject(
                            name='?q',
                            type_name='object'
                        )
                ] + [TypedObject(name=f'?X_W{other}', type_name='object') for other in other_objects]
                preconds = [
                        Atom('clear', (other_object_pddl,)),
                        Atom('ik', (arm_pddl, object_pddl,'?X_WB', '?X_HB', '?pre_q', '?q')),
                        Atom('athandpose', (arm_pddl, object_pddl, '?X_HB')),
                        # Atom('atconf', (arm_pddl, '?pre_q')),
                        Atom('block-support', (object_pddl, '?X_WB', other_object_pddl, f"?X_W{other_object}")),
                        Atom('atworldpose', (other_object_pddl, f"?X_W{other_object}")),
                        Atom('grasped', (arm_pddl, object_pddl)),
                ]
                for other in other_objects:
                    if other == other_object:
                        continue
                    preconds.append(Atom('atworldpose', (Object.from_value(other).pddl, f"?X_W{other}")))
                    preconds.append(Atom('colfree-block', (arm_pddl, '?q', Object.from_value(other).pddl, f"?X_W{other}")))
    
                effects = [
                    Atom('empty', (arm_pddl,)),
                    Atom('on-block', (object_pddl, other_object_pddl)),
                    Atom('atworldpose', (object_pddl, '?X_WB')),
                    Atom('athandpose', (arm_pddl, object_pddl, '?X_HB')).negate(),
                    Atom('clear', (other_object_pddl,)).negate(),
                    Atom('grasped', (arm_pddl, object_pddl)).negate(),
                ]
                actions.append(
                    Action(
                        name=f"stack({arm},{object},{other_object})",
                        parameters=params,
                        num_external_parameters=len(params),
                        precondition=Conjunction(preconds),
                        effects=[Effect([], conditions.Truth(), atom) for atom in effects],
                        cost=1,
                    )
                )
        # # MOVE #
        # params = [
        #         TypedObject(
        #             name='?q1',
        #             type_name='object'
        #         ),
        #         TypedObject(
        #             name='?traj',
        #             type_name='object'
        #         ),
        #         TypedObject(
        #             name='?q2',
        #             type_name='object'
        #         ),
        # ]
        # preconds = [
        #         Atom('motion', (arm_pddl, '?q1', '?traj', '?q2')),
        #         Atom('atconf', (arm_pddl, '?q1')),
        # ]

        # effects = [
        #     Atom('atconf', (arm_pddl, '?q1')).negate(),
        #     Atom('atconf', (arm_pddl, '?q2')),
        # ]
        # actions.append(
        #     Action(
        #         name=f"move({arm})",
        #         parameters=params,
        #         num_external_parameters=len(params),
        #         precondition=Conjunction(preconds),
        #         effects=[Effect([], conditions.Truth(), atom) for atom in effects],
        #         cost=1,
        #     )
        # )

    return actions


if __name__ == '__main__':

    problem_file = 'experiments/blocks_world/data_generation/easy_distractors/test/3_23_2_97.yaml'
    init, goal, externals, actions = create_problem(problem_file)

    print('Initial:', init)
    print('\n\nGoal:', goal)
    from lifted.a_star import ActionStreamSearch, repeated_a_star

    search = ActionStreamSearch(init, goal, externals, actions)
    stats = {}

    def heuristic(state, goal):
        # actions = [a for _, a, _ in state.get_shortest_path_to_start()]
        # if len(actions) >= 2:
        #     if actions[-1] is not None and "move" in actions[-1].name:
        #         if actions[-2] is not None and "move" in actions[-2].name:
        #             return np.inf
        return len(goal - state.state)

    profile = 'lifted_2nonmono.profile'
    if profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
    result = repeated_a_star(search, stats=stats, max_steps=10, heuristic=heuristic)
    if result.solution is not None:
        action_skeleton, object_mapping = result.action_skeleton, result.object_mapping
        actions_str = "\n".join([str(a) for a in action_skeleton])
        print(f"Action Skeleton:\n{actions_str}")
        print(f"\nObject mapping: {object_mapping}\n") 
        print("Expanded", result.expand_count)
    if profile:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.print_stats()
            ps.dump_stats(profile)   

# %%
