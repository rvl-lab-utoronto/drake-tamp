#!/usr/bin/env python3
"""
Simple script to make cubes
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

# SIZE = "0.045 0.045 0.045"
TEMPLATE_NAME = "template_hand_and_block.sdf"
FILE_PATH, _ = os.path.split(os.path.realpath(__file__))


def make_hand_and_block(name, panda_color, block_color):
    tree = ET.parse(f"{FILE_PATH}/{TEMPLATE_NAME}")
    root = tree.getroot()
    for model in root.iter("model"):
        model.set("name", name)

    model = root[0]
    for link in model:
        if link.tag != "link":
            continue
        link_name = link.attrib.get("name", "")
        if link_name.startswith("panda"):
            for diffuse in link.iter("diffuse"):
                diffuse.text = panda_color
        if link_name == "base_link":
            for diffuse in link.iter("diffuse"):
                diffuse.text = block_color

    return tree

def random_color():
    r, g, b = np.random.uniform(low =0, high = 1, size = 3)
    return f"{r} {g} {b} 1"

if __name__ == "__main__":

    name = "test_hand_and_block"

    for i, green in enumerate(np.linspace(0,1,10)):
        tree = make_hand_and_block(name, panda_color = f'0 {green} 0 1', block_color = random_color())
        tree.write(f"hand_and_block_{i}" + ".sdf")
