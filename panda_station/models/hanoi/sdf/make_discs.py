#!/usr/bin/env python3
"""
Simple script to make discs
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

TEMPLATE_NAME = "template_disc.sdf"
FILE_PATH, _ = os.path.split(os.path.realpath(__file__))

def make_disc(name, color, size):
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
        if name == "visual":
            r = size[0]
            l = size[1]
            z = l/2
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("cylinder").find("radius").text = f"{r}"
            elm.find("geometry").find("cylinder").find("length").text = f"{l}"
        elif name == "collision":
            r = size[0]
            l = size[1]
            z = l/2
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("cylinder").find("radius").text = f"{r}"
            elm.find("geometry").find("cylinder").find("length").text = f"{l}"
    return tree


if __name__ == "__main__":

    min_r = 0.01
    max_r = 0.036
    inc_r = 0.002
    length = 0.02
    for r in np.arange(min_r, max_r + inc_r/2, inc_r):
        name = f"disc_{int(r*1000)}"
        tree = make_disc(name, f"0 0 {0.3 + 0.7*(max_r-r)/(max_r - min_r)} 1", np.array([r, length]))
        tree.write(name + ".sdf")
