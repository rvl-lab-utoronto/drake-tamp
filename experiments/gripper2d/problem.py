"""This experiment looks at a very simple domain where scenarios only differ by the placement of a fixed number (2) of blocks which
have fixed sizes. We want to see if it is possible to use supervised learning to predict the feasibility of picking up each of the
objects, given their placements."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from frozendict import frozendict
from copy import deepcopy
import imageio

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

def check_overlap(l1, r1, l2, r2):
    # To check if either rectangle is actually a line
    # For example :  l1 ={-1,0}  r1={1,1}  l2={0,-1}
    # r2={0,1}
 
    if (l1[0] == r1[0] or l1[1] == r1[1] or l2[0] == r2[0]
        or l2[1] == r2[1]):
        # the line cannot have positive overlap
        return False
 
    # If one rectangle is on left side of other
    if (l1[0] >= r2[0] or l2[0] >= r1[0]):
        return False
 
    # If one rectangle is above other
    if (r1[1] >= l2[1] or r2[1] >= l1[1]):
        return False
 
    return True

def check_safe(gripper, gripperpose, block, blockpose):
    return not check_overlap(
        (gripperpose[0], gripperpose[1] + gripper["height"]),
        (gripperpose[0] + gripper["width"], gripperpose[1]),
        (blockpose[0], blockpose[1] + block["height"]),
        (blockpose[0] + block["width"], blockpose[1]),
    )

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


def visualize(world, grippers, regions, blocks, path=None):
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
    if path is not None:
        plt.savefig(path, dpi=300)


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


def generate_scene(block_heights=None, region_widths=None, height=10):

    if block_heights is None:
        num_blocks = np.random.randint(2, 5)
        heights = [[np.random.randint(1, 4) for block in range(num_blocks)]]

    region_widths = [5, 1, 4] if region_widths is None else region_widths
    
    WORLD = {
        "height": height,
        "width": sum(region_widths),
    }    
    
    REGIONS = frozendict({
        f"r{int(i/2)+1}": {"width": w, "x": sum(region_widths[:i]), "y": -0.3, "height": .3}
        for i, w in enumerate(region_widths) if i%2 == 0
    })

    GRIPPERS = {
        "g1": {"width": 1.5, "height": 1, "x": 2, "y": 8, "color": None}
    }


    colors = plt.get_cmap("tab10")
    
    BLOCKS = {}
    block_counter = 0
    for i, regional_block_heights in enumerate(block_heights):
        region = f"r{i+1}"
        _BLOCKS = {
            f"b{block_counter + j}": {
                "width": 1,
                "y": 0,
                "height": height,
                "color": colors(block_counter + j),
                "on": region
            } for j, height in enumerate(regional_block_heights)
        }
        block_counter += len(regional_block_heights)
        set_placements(REGIONS[region], _BLOCKS)
        BLOCKS.update(_BLOCKS)

    return (WORLD, GRIPPERS, REGIONS, BLOCKS)

def visualize_plan(scene, objects, action_skeleton, object_mapping, path=None):
    visualize(*scene, path=f"{path}0.png" if path is not None else None)
    world, grippers, regions, blocks = deepcopy(scene)
    for i,action in enumerate(action_skeleton[:]):
        if 'pick' in action.name:
            block = action.name.split(',')[1]
            grippers['g1']['x'], grippers['g1']['y'] = object_mapping[action.var_mapping[f'?conf']].value
        elif 'place' in action.name:
            block = action.name.split(',')[1]
            print(block)
            grippers['g1']['x'], grippers['g1']['y'] = object_mapping[action.var_mapping[f'?conf']].value
            blocks[block]['x'], blocks[block]['y'] = object_mapping[action.var_mapping[f'?blockpose']].value
        visualize(world, grippers, regions, blocks, path=f"{path}{i + 1}.png" if path is not None else None)
        
    if path is not None:
        images = []
        filenames = [f'{path}{i}.png' for i in range(len(action_skeleton))]
        print(filenames)
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{path}.gif', images, duration=1)
