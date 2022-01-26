from learning.cg_toy_simple import ancestral_sampling_acc
from lifted_search import StreamAction
from learning.cg_num_obects import generate_scene, scene_objects, ik, check_safe, grasp, placement, set_placements, WORLD, np, plt, visualize
from pddlstream.language.stream import Stream, StreamInfo
from pddlstream.language.object import Object, OptimisticObject
from pddlstream.language.generator import from_gen, from_gen_fn, from_test

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

def pick_place_cg(scene, objects, block_name, source='r1', to='r2', suffix='', object_to_pose_pddl={}):
    place_id = f'{block_name}_on_{to}{suffix}'
    grasp_id = f'{block_name}_grasp_{source}{suffix}'

    return [
        StreamAction(grasp_stream, (objects['g1'].pddl, objects[block_name].pddl), (f'?{grasp_id}',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[block_name].pddl, object_to_pose_pddl[block_name], f'?{grasp_id}'), (f'?{grasp_id}conf',)),
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, f'?{grasp_id}conf', objects[bname].pddl, object_to_pose_pddl[bname]), (f'?safe_{bname}_{grasp_id}conf',))
        for bname in scene[-1] if bname != block_name
    ] + [
        StreamAction(placement_stream, (objects[block_name].pddl, objects[to].pddl), (f'?{place_id}',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[block_name].pddl, f'?{place_id}', f'?{grasp_id}'), (f'?{place_id}conf',))
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, f'?{place_id}conf', objects[bname].pddl, object_to_pose_pddl[bname]), (f'?safe_{bname}_{place_id}conf',))
        for bname in scene[-1] if bname != block_name
    ] + [
        StreamAction(safe_block_stream, (objects[block_name].pddl, f'?{place_id}', objects[bname].pddl, object_to_pose_pddl[bname]), (f'?blocksafe_{bname}_{place_id}',))
        for bname in scene[-1] if bname != block_name
    ]

def sequence_pick_place_cg(scene, objects, blocks, sources, destinations):
    object_to_pose_pddl = {b:objects[f'{b}_pose'].pddl for b in scene[-1]}
    ordering = []
 
    for i, (block, source, dest) in enumerate(zip(blocks, sources, destinations)):
        ordering.extend(
            pick_place_cg(scene, objects, block, source, dest, suffix=f"{i}", object_to_pose_pddl=object_to_pose_pddl)
        )
        object_to_pose_pddl[block] = f'?{block}_on_{dest}{i}'
    
    assert len([o for action in ordering for o in action.outputs]) == len(set([o for action in ordering for o in action.outputs]))
    return ordering


if __name__ == '__main__':
    import time
    import pprint
    np.random.seed(42)
    block_name = 'b0'
    scene = generate_scene()
    objects = scene_objects(*scene)
    ordering = sequence_pick_place_cg(scene, objects, [block_name, block_name], ['r1', 'r2'], ['r2', 'r1'])
    pprint.pprint(ancestral_sampling_acc(objects, ordering, 100))

    # All the different two step moves:
    # There are N objects, and M regions
    # For the first move, I could pick 
    # any of the N objects, to move to any
    # of the M regions.
    # For the second move, the situation is
    # the same. So we get: (N^2 * M^2)
    orderings = []
    plans = []
    for block in scene[-1]:
        for region in scene[-2]:
            for block2 in scene[-1]:
                for region2 in scene[-2]:
                    blocks = [block]
                    sources = ['r1']
                    dests = [region]
                    blocks.append(block2)
                    sources.append('r1' if block2 != block else region)
                    dests.append(region2)
                    plan = list(zip(blocks, sources, dests))
                    plans.append(plan)
                    ordering = sequence_pick_place_cg(scene, objects, blocks, sources, dests)
                    orderings.append(ordering)
    
    stats = []
    for plan, ordering in zip(plans, orderings):
        stats.append(ancestral_sampling_acc(objects, ordering, 100))
    
        print(plan, stats[-1].get('FINAL', 0) / stats[-1]['START'])