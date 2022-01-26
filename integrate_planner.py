#%%
import sys
sys.path.insert(0, '/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
sys.path.insert(0, '/home/atharv/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
import pddl
from pddl.conditions import Atom, Conjunction, NegatedAtom
from pddl.actions import PropositionalAction, Action
from pddl.effects import Effect
from pddl.pddl_types import TypedObject
import pddl.conditions as conditions


from learning.cg_num_obects import generate_scene, scene_objects, ik, check_safe, grasp, placement, set_placements, WORLD, np, plt, visualize

WORLD['width'] = 20

def generate_scene(heights=None):
    if heights is None:
        num_blocks = np.random.randint(2, 5)
        heights = [np.random.randint(1, 4) for block in range(num_blocks)]

    REGIONS = {
        "r1": {"width": 5, "x": 0, "y": -0.3, "height": .3},
        "r2": {"width": 4, "x": 6, "y": -0.3, "height": .3},
        "r3": {"width": 4, "x": 11, "y": -0.3, "height": .3},
        "r4": {"width": 4, "x": 16, "y": -0.3, "height": .3},
        
    }

    GRIPPERS = {
        "g1": {"width": 1.8, "height": 1, "x": 2, "y": 8, "color": None}
    }


    colors = plt.get_cmap("tab10")
    BLOCKS = {
        f"b{i}": {
            "width": 1.2,
            "y": 0,
            "height": height,
            "color": colors(i)
        } for i, height in enumerate(heights)
    }
    set_placements(REGIONS['r1'], BLOCKS)

    return (WORLD, GRIPPERS, REGIONS, BLOCKS)

from pddlstream.language.stream import Stream, StreamInfo
from pddlstream.language.object import Object, OptimisticObject
from pddlstream.language.generator import from_gen, from_gen_fn, from_test

stream_info = StreamInfo(use_unique=True)
grasp_stream = Stream(
    'grasp',
    from_gen_fn(grasp),
    ('?gripper', '?block'),
    [('gripper', '?gripper'), ('block', '?block')],
    ('?grasp',),
    [('grasp', '?gripper', '?block', '?grasp')],
    info=stream_info
)
placement_stream = Stream(
    'placement',
    from_gen_fn(placement),
    ('?block', '?region'),
    [('block', '?block'), ('region', '?region')],
    ('?blockpose',),
    [('placement', '?block', '?region', '?blockpose'), ('blockpose', '?blockpose')],
    info=stream_info
)
ik_stream = Stream(
    'ik',
    from_gen_fn(lambda g,_,bp,gr: ik(g, bp, gr)),
    ('?gripper', '?block', '?blockpose', '?grasp'),
    [('gripper', '?gripper'), ('grasp', '?gripper', '?block', '?grasp'), ('blockpose', '?blockpose')],
    ('?conf',),
    [('ik', '?gripper', '?block','?conf', '?blockpose', '?grasp'), ('conf', '?conf')],
    info=stream_info
)
safety_stream = Stream(
    'safe',
    from_test(check_safe),
    ('?gripper', '?gripperpose', '?block', '?blockpose'),
    [('gripper', '?gripper'), ('conf', '?gripperpose'), ('block', '?block'), ('blockpose', '?blockpose')],
    tuple(),
    [('safe', '?gripper', '?gripperpose', '?block', '?blockpose')],
    info=stream_info
)
safe_block_stream = Stream(
    'safe-block',
    from_test(check_safe),
    ('?block1', '?blockpose1', '?block', '?blockpose'),
    [('block', '?block1'), ('blockpose', '?blockpose1'), ('block', '?block'), ('blockpose', '?blockpose')],
    tuple(),
    [('safe-block', '?block1', '?blockpose1', '?block', '?blockpose')],
    info=stream_info
)

externals = [
    grasp_stream,
    ik_stream,
    placement_stream,
    safety_stream,
    safe_block_stream
]
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
    
    return objects, actions, initial_state
if __name__ == '__main__':
    from lifted_search import ActionStreamSearch, repeated_a_star, find_applicable_brute_force
#%%
    import time
    np.random.seed(int(time.time()))
    scene = generate_scene([1])

    world, grippers, regions, blocks = scene
    objects, actions, initial_state = instantiate_planning_problem(scene)
#%%
    goal = set()
    # goal.add(Atom('holding', (objects['g1'].pddl, objects['b0'].pddl)))
    goal.add(Atom('on', (objects['b0'].pddl, objects['r2'].pddl)))
    # goal.add(Atom('on', (objects['b1'].pddl, objects['r1'].pddl)))
    # goal.add(Atom('on', (objects['b2'].pddl, objects['r1'].pddl)))
    # goal.add(Atom('on', (objects['b3'].pddl, objects['r1'].pddl)))

#%%
    start = time.time()
    search = ActionStreamSearch(initial_state, goal, externals, actions)
    stats = {}
    result = repeated_a_star(search, stats=stats, max_steps=10)
    if result is not None:
        action_skeleton, object_mapping, _ = result
        actions_str = "\n".join([str(a) for a in action_skeleton])
        print(f"Action Skeleton:\n{actions_str}")
        print(f"\nObject mapping: {object_mapping}\n") 
    print('Took', time.time() - start, 'seconds')
#%%
    start = time.time()
    search = ActionStreamSearch(initial_state, goal, externals, actions)
    result = repeated_a_star(search, stats=stats)
    assert result is not None
    print('With stats took', time.time() - start, 'seconds')

#%%
    from lifted_search import StreamAction
    from learning.cg_num_obects import ancestral_sampling_acc
    obstruction = 'b1'
    block_name = 'b0'
    plan_prefix = [
        StreamAction(grasp_stream, (objects['g1'].pddl, objects[obstruction].pddl), ('?grasp',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[obstruction].pddl, objects[obstruction + '_pose'].pddl, '?grasp'), ('?conf',)),
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}_pick1',))
        for bname in scene[-1] if bname != obstruction
    ] + [
        StreamAction(placement_stream, (objects[obstruction].pddl, objects['r2'].pddl), ('?obsplace',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[obstruction].pddl, '?obsplace', '?grasp'), ('?obsplaceconf',))
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?obsplaceconf', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}_place1',))
        for bname in scene[-1] if bname != obstruction
    ] + [
        StreamAction(safe_block_stream, (objects[obstruction].pddl, '?obsplace', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?blocksafe_{bname}_place1',))
        for bname in scene[-1] if bname != obstruction
    ] 

    prefix_stats = ancestral_sampling_acc(objects, plan_prefix, 200)
    # %%

    plan = plan_prefix + [
        StreamAction(grasp_stream, (objects['g1'].pddl, objects[block_name].pddl), ('?grasp2',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[block_name].pddl, objects[block_name + '_pose'].pddl, '?grasp2'), ('?conf2',)),
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf2', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}_pick2',))
        for bname in scene[-1] if bname not in [block_name, obstruction]
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf2', objects[obstruction].pddl, '?obsplace'), (f'?safe_{obstruction}_pick2',)) # the new place of obstruction
    ] + [
        StreamAction(placement_stream, (objects[block_name].pddl, objects['r2'].pddl), ('?bplace',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[block_name].pddl, '?bplace', '?grasp2'), ('?bplaceconf',))
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?bplaceconf', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}_place2',))
        for bname in scene[-1] if bname not in [block_name, obstruction]
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?bplaceconf', objects[obstruction].pddl, '?obsplace'), (f'?safe_{obstruction}_place2',)) # the new place of obstruction
    ] + [
        StreamAction(safe_block_stream, (objects[block_name].pddl, '?bplace', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?blocksafe_{bname}_place2',))
        for bname in scene[-1] if bname not in [block_name, obstruction]
    ] + [
        StreamAction(safe_block_stream, (objects[block_name].pddl, '?bplace', objects[obstruction].pddl, '?obsplace'), (f'?blocksafe_{obstruction}_place2',)) # the new place of obstruction
    ]

    stats = ancestral_sampling_acc(objects, plan, 200)
    # %%
    for node in prefix_stats:
        print(stats[node], prefix_stats[node])


    # %%
    from learning.cg_num_obects import get_stream_action_edges
    from pyvis.network import Network
    from  matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('rg',["r", "b", "g"], N=stats['START']) 
    net = Network(directed=True, width="100%", height="100%", layout="hierarchy", font_color="white")
    net.repulsion(
        node_distance=600, spring_strength=0, spring_length=600, central_gravity=0
    )
    nodes = plan
    edges = get_stream_action_edges(plan)
    adjacency = {}
    for (e,s) in edges:
        adjacency.setdefault(s, set()).add(e)

    def get_level(node):
        if node not in adjacency:
            return 0
        else:
            return 1 + max(get_level(parent) for parent in adjacency[node])

    max_level = 0
    for node in plan + ['FINAL']:
        level = get_level(node) if node != 'FINAL' else max_level + 1
        max_level = max(level, max_level)
        r,g,b,a = [255 * c for c in cmap(stats[node])]
        label = str((node.stream.name, node.inputs)) if node != 'FINAL' else node
        net.add_node(hash(node), label=label, level=level, node_repulsion=True, shape="box", color=f"rgba({r},{g},{b},1)")
    for edge in edges:
        net.add_edge(hash(edge[0]), hash(edge[1]))

    for node in plan:
        net.add_edge(hash(node), hash('FINAL'))


    net.set_options(
        """
    var options = {
    "layout": {
        "hierarchical": {
        "enabled": true
        }
    },
    "physics": {
        "hierarchicalRepulsion": {
        "centralGravity": 0,
        "nodeDistance": 270
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
    }
    }
        """ 
    )
    # net.show(name = "graph.html")
    # net.show_buttons(filter_=["physics", "layout"])
    net.save_graph('list_of_nodes.html')
        # %%
#%%
    original_scene = tuple(x.copy() for x in scene)
    # %%
    obstruction = 'b1'
    spots = []
    for i in range(1000):
        while True:
            try:
                # pick obstruction
                (grasp_ob,) = next(grasp(grippers['g1'], blocks[obstruction]))
                (conf, ) = next(ik(grippers['g1'], objects[f'{obstruction}_pose'].value,  grasp_ob))
                safe = True
                for block in blocks:
                    if block != obstruction:
                        safe = safe and check_safe(grippers['g1'], objects[f'g1_pose'].value, blocks[block], objects[f'{block}_pose'].value)
                if not safe:
                    continue


                # place obstruction
                (obsplace, ) = next(placement(blocks[obstruction], regions['r2']))
                (obsplaceconf, ) = next(ik(grippers['g1'], obsplace,  grasp_ob))
                break
            except StopIteration:
                continue
        spots.append(obsplace)
        

    # %%
    world, grippers, regions, blocks = scene
    blocks = blocks.copy()
    blocks[obstruction]['x'], blocks[obstruction]['y'] = spots[np.random.randint(len(spots))]
    visualize(world, grippers, regions, blocks)
    new_scene = (world, grippers, regions, blocks)

    # %%
    objects = scene_objects(*new_scene)
    the_block = 'b0'
    attempts = 2000
    pick_worked = place_ik_worked = pick_ik_worked = success = 0
    for i in range(attempts):
        try:
            # pick the_block
            (grasp_ob,) = next(grasp(grippers['g1'], blocks[the_block]))
            # print(grasp_ob + objects[f'{the_block}_pose'].value)
            (conf, ) = next(ik(grippers['g1'], objects[f'{the_block}_pose'].value,  grasp_ob))

            pick_ik_worked += 1
            safe = True
            for block in blocks:
                if block != the_block:
                    safe = safe and check_safe(grippers['g1'], objects[f'g1_pose'].value, blocks[block], objects[f'{block}_pose'].value)
            if not safe:
                continue

            pick_worked += 1
            # place the_block
            (obsplace, ) = next(placement(blocks[the_block], regions['r2']))
            (obsplaceconf, ) = next(ik(grippers['g1'], obsplace,  grasp_ob))
            place_ik_worked += 1
            for block in blocks:
                if block != the_block:
                    _safe = check_safe(grippers['g1'], obsplaceconf, blocks[block], objects[f'{block}_pose'].value)
                    # if not _safe:
                    #     print(block, 'obstructing gripper')
                    safe = safe and _safe
                    _safe = check_safe(blocks[the_block], obsplace, blocks[block], objects[f'{block}_pose'].value)
                    # if not _safe:
                    #     print(block, 'obstructing', the_block)
                    safe = safe and _safe
                    
            if not safe:
                # grips = grippers.copy()
                # bs = blocks.copy()
                # grips['g1']['x'], grips['g1']['y'] = obsplaceconf
                # bs[the_block]['x'], bs[the_block]['y'] = obsplace
                # # plt.figure()
                # visualize(world, grips, regions, bs)
                # break
                continue
            success += 1
        except StopIteration:
            continue

    print(pick_ik_worked)
    print(pick_worked)
    print(place_ik_worked)
    print(success)
    # %%
    objects = scene_objects(*new_scene)
    obstruction = 'b0'
    # block_name = 'b0'
    plan_prefix = [
        StreamAction(grasp_stream, (objects['g1'].pddl, objects[obstruction].pddl), ('?grasp',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[obstruction].pddl, objects[obstruction + '_pose'].pddl, '?grasp'), ('?conf',)),
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}_pick1',))
        for bname in scene[-1] if bname != obstruction
    ] + [
        StreamAction(placement_stream, (objects[obstruction].pddl, objects['r2'].pddl), ('?obsplace',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[obstruction].pddl, '?obsplace', '?grasp'), ('?obsplaceconf',))
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?obsplaceconf', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}_place1',))
        for bname in scene[-1] if bname != obstruction
    ] + [
        StreamAction(safe_block_stream, (objects[obstruction].pddl, '?obsplace', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?blocksafe_{bname}_place1',))
        for bname in scene[-1] if bname != obstruction
    ] 

    prefix_stats = ancestral_sampling_acc(objects, plan_prefix, 2000)
    print(prefix_stats['FINAL'])
    # %%
    # %%
    successes = []
    for spot in spots:
        world, grippers, regions, blocks = scene
        blocks = blocks.copy()
        blocks[obstruction]['x'], blocks[obstruction]['y'] = spot
        new_scene = (world, grippers, regions, blocks)

        objects = scene_objects(*new_scene)
        the_block = 'b0'
        attempts = 100
        pick_worked = place_ik_worked = pick_ik_worked = success = 0
        for i in range(attempts):
            try:
                # pick the_block
                (grasp_ob,) = next(grasp(grippers['g1'], blocks[the_block]))
                # print(grasp_ob + objects[f'{the_block}_pose'].value)
                (conf, ) = next(ik(grippers['g1'], objects[f'{the_block}_pose'].value,  grasp_ob))

                pick_ik_worked += 1
                safe = True
                for block in blocks:
                    if block != the_block:
                        safe = safe and check_safe(grippers['g1'], objects[f'g1_pose'].value, blocks[block], objects[f'{block}_pose'].value)
                if not safe:
                    continue

                pick_worked += 1
                # place the_block
                (obsplace, ) = next(placement(blocks[the_block], regions['r2']))
                (obsplaceconf, ) = next(ik(grippers['g1'], obsplace,  grasp_ob))
                place_ik_worked += 1
                for block in blocks:
                    if block != the_block:
                        _safe = check_safe(grippers['g1'], obsplaceconf, blocks[block], objects[f'{block}_pose'].value)
                        # if not _safe:
                        #     print(block, 'obstructing gripper')
                        safe = safe and _safe
                        _safe = check_safe(blocks[the_block], obsplace, blocks[block], objects[f'{block}_pose'].value)
                        # if not _safe:
                        #     print(block, 'obstructing', the_block)
                        safe = safe and _safe
                        
                if not safe:
                    # grips = grippers.copy()
                    # bs = blocks.copy()
                    # grips['g1']['x'], grips['g1']['y'] = obsplaceconf
                    # bs[the_block]['x'], bs[the_block]['y'] = obsplace
                    # # plt.figure()
                    # visualize(world, grips, regions, bs)
                    # break
                    continue
                success += 1
            except StopIteration:
                continue

        successes.append((pick_ik_worked,pick_worked, place_ik_worked, success))

    # %%
    def cost(state, op, child, verbose=False):
        included = set()
        c = 0
        for obj in child.object_stream_map:
            stream_action = child.object_stream_map[obj]
            if stream_action is not None:
                if stream_action in included:
                    continue
                cg_key = child.get_object_computation_graph_key(obj)
                if cg_key in stats:
                    s = stats[cg_key]
                    comp_cost = ((s['num_successes'] + 1) / (s['num_attempts'] + 1))**-1
                    c += comp_cost

                else:
                    comp_cost = 1
                    c += comp_cost
                if verbose:
                    print(stream_action, comp_cost)
                included.add(stream_action)
        return max(1, c)

    state = search.init.children[2][1].children[2][1]
    for (op, child) in state.children:
        print(op, cost(None, None, child, verbose=True))
    # %%
