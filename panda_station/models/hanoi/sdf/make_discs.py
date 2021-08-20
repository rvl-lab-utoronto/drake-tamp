#!/usr/bin/env python3
"""
Simple script to make discs
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

TEMPLATE_NAME = "template_disc.sdf"
FILE_PATH, _ = os.path.split(os.path.realpath(__file__))

def make_disc(name, color, size, ball_radius = 1e-7):
    tree = ET.parse(f"{FILE_PATH}/{TEMPLATE_NAME}")
    root = tree.getroot()
    for model in root.iter("model"):
        model.set("name", name)
    for diffuse in root.iter("diffuse"):
        diffuse.text = color

    model = root[0]
    base = model[0]
    r = size[0]
    l = size[1]
    z = l/2
    for elm in base:
        name = elm.attrib.get("name", "")
        if name.startswith("ball"):
            num = int(name[-1]) - 1
            x = (-1) ** (num & 0b01) * r/np.sqrt(2)
            y = (-1) ** ((num & 0b10) >> 1) * r/np.sqrt(2)
            elm.find("pose").text = f"{x} {y} {0} 0 0 0"
            elm.find("geometry").find("sphere").find("radius").text = f"{ball_radius}"
        elif name == "visual":
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("cylinder").find("radius").text = f"{r}"
            elm.find("geometry").find("cylinder").find("length").text = f"{l}"
        elif name == "collision":
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("cylinder").find("radius").text = f"{r}"
            elm.find("geometry").find("cylinder").find("length").text = f"{l}"
    return tree


if __name__ == "__main__":

    min_r = 0.01
    max_r = 0.036
    inc_r = 0.001
    length = 0.02
    for r in np.arange(min_r, max_r + inc_r/2, inc_r):
        name = f"disc_{int(r*1000)}"
        red,green,blue = np.random.uniform(0, 1, size = (3,))
        tree = make_disc(name, f"{red} {green} {blue} 1", np.array([r, length]))
        tree.write(name + ".sdf")
