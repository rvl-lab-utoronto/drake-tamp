"""This experiment looks at a very simple domain where scenarios only differ by the placement of a fixed number (2) of blocks which
have fixed sizes. We want to see if it is possible to use supervised learning to predict the feasibility of picking up each of the
objects, given their placements."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from frozendict import frozendict

def grasp(g, b):
    # relative to the bottom left of the block
    # has to be within 1e-3 from the top of the block
    # has to contact at least half of the block width
    if g['width'] < b['width'] / 2:
        return
    xmin = -(g['width'] - b['width'] / 2)
    xmax = b['width'] / 2
    xrange = xmax - xmin
    while True:
        y = b['height'] + 1e-3 * np.random.random()
        x = xrange*np.random.random() + xmin
        yield (np.array([x, y]), )

def ik(gripper, blockpose, grasp, world):
    # TODO: This is actually wrong. Need to 
    gripperpose = grasp + blockpose
    if np.any(np.array(gripperpose) < 0):
        return
    if np.any(np.array(gripperpose) + [gripper['width'], gripper['height']] > [world['width'], world['height']]):
        return
    yield (gripperpose, )

def _check_safe(gripper, gripperpose, block, blockpose):
    corners = [
        [blockpose[0], blockpose[1]],
        [blockpose[0] + block['width'], blockpose[1]],
        [blockpose[0], blockpose[1] + block['height']],
        [blockpose[0] + block['width'], blockpose[1] + block['height']]
    ]
    for corner in corners:
        if (gripperpose[0] <= corner[0] <= gripperpose[0] + gripper['width']) and \
        (gripperpose[1] <= corner[1] <= gripperpose[1] + gripper['height']):
            return False
    return True
def check_safe(gripper, gripperpose, block, blockpose):
    return _check_safe(gripper, gripperpose, block, blockpose) and _check_safe(block, blockpose, gripper, gripperpose)


def placement(block, region):
    if block['width'] > region['width']:
        return

    y = region['y'] + region['height']
    xmin = region['x']
    xmax = region['x'] + region['width'] - block['width']
    xrange = xmax - xmin
    while True:
        x = xrange*np.random.random() + xmin
        yield (np.array([x, y]), )

def visualize(world, grippers, regions, blocks):
    fig = plt.figure()
    ax = plt.gca()
    plt.xlim(0, world['width'])
    plt.ylim(-.5, world["height"])
    
    region = Rectangle([0, -0.5], world['width'], 0.2, color='black')
    ax.add_patch(region)
#     region = Rectangle([-.2, -.5], .2, world['height'] + .2*2, color='black')
#     ax.add_patch(region)
#     region = Rectangle([world['width'], -.5], .2, world['height'] + .2*2, color='black')
#     ax.add_patch(region)
        
    for rname in regions:
        r = regions[rname]
        region = Rectangle([r['x'], r['y']], r['width'], r['height'], alpha=0.8)
        ax.add_patch(region)

    for bname in blocks:
        r = blocks[bname]
        block = Rectangle([r['x'], r['y']], r['width'], r['height'], color=r['color'], alpha=0.8)
        ax.add_patch(block)
        
    for gname in grippers:
        r = grippers[gname]
        gripper = Rectangle([r['x'], r['y']], r['width'], r['height'], color=r['color'], alpha=0.8)
        ax.add_patch(gripper)
        stem_width = r['width'] / 4
        stem_height = world['height'] - (r['y'] + r['height'])
        stem = Rectangle([(2*r['x'] + r['width']) / 2 - stem_width / 2, (r['y'] + r['height'])], stem_width, stem_height, color=r['color'], alpha=0.8)
        ax.add_patch(stem)

def random(start, end):
    return (np.random.random() * (end - start)) + start

def set_placements(region, blocks):
    
    start = region['x']
    end = region['x'] + region['width']
    remaining = end - start
    min_required = sum(b['width'] for b in blocks.values()) + 1e-2

    if min_required > remaining:
        raise ValueError((min_required, remaining))

    block_names = list(blocks)
    np.random.shuffle(block_names)

    for block in block_names:
        blocks[block]['x'] = random(start, end - min_required)
        start = blocks[block]['x'] + blocks[block]['width']
        min_required = min_required - blocks[block]['width']


def generate_scene(heights=None):
    WORLD = {
        "height": 10,
        "width": 10,
    }
    if heights is None:
        num_blocks = np.random.randint(2, 5)
        heights = [np.random.randint(1, 4) for block in range(num_blocks)]

    REGIONS = frozendict({
        "r1": {"width": 5, "x": 0, "y": -0.3, "height": .3},
        "r2": {"width": 4, "x": 6, "y": -0.3, "height": .3}
    })

    GRIPPERS = {
        "g1": {"width": 1.5, "height": 1, "x": 2, "y": 8, "color": None}
    }


    colors = plt.get_cmap("tab10")
    BLOCKS = {
        f"b{i}": {
            "width": 1,
            "y": 0,
            "height": height,
            "color": colors(i)
        } for i, height in enumerate(heights)
    }
    set_placements(REGIONS['r1'], BLOCKS)

    return (WORLD, GRIPPERS, REGIONS, BLOCKS)
