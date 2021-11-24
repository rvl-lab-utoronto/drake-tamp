#!/usr/bin/env python3
"""
Simple script to make cubes
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

TEMPLATE_NAME = "template_block.sdf"
FILE_PATH, _ = os.path.split(os.path.realpath(__file__))

def make_cube(name, color, size, buffer, ball_radius=1e-7):
    tree = ET.parse(f"{FILE_PATH}/{TEMPLATE_NAME}")
    root = tree.getroot()
    for model in root.iter("model"):
        model.set("name", name)
    for diffuse in root.iter("diffuse"):
        diffuse.text = color

    model = root[0]
    base = model[0]
    for elm in base:
        name = elm.attrib.get("name", "")
        if name.startswith("ball"):
            num = int(name[-1]) - 1
            x = (-1) ** (num & 0b01) * size[0] / 2
            y = (-1) ** ((num & 0b10) >> 1) * size[1] / 2
            z = ((num & 0b100) >> 2) * size[2]
            elm.find("pose").text = f"{x} {y} {z} 0 0 0"
            elm.find("geometry").find("sphere").find("radius").text = f"{ball_radius}"
        elif name == "visual":
            w = size[0]
            d = size[1]
            h = size[2]
            z = h/2
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("box").find("size").text = f"{w} {d} {h}"
        elif name == "collision":
            w = size[0] - buffer*2
            d = size[1] - buffer*2
            h = size[2] - buffer*2
            z = size[2]/2
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("box").find("size").text = f"{w} {d} {h}"

    return tree


if __name__ == "__main__":

    block_size = np.array([0.05, 0.05, 0.05])
    blocker_size = np.array([0.05, 0.05, 0.1])
    buffer = 0.001
    colors = [
        "1 0.1 0.1 1",
        "0.956 1 0.1 1",
        "0.1 0.1 1 1",
        "1 0.239 0.474 1",
        "0.4157 0.2863 0.251 1",
    ]
    infos = [
        ("red_block", block_size),
        ("yellow_block", block_size),
        ("indigo_block", block_size),
        ("pink_block", block_size),
        ("wood_block", block_size),
    ]

    for info, color in zip(infos, colors):
        name, size = info
        tree = make_cube(name, color, size, buffer)
        tree.write(name + ".sdf")
