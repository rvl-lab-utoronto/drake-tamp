#!/usr/bin/env python3
"""
Simple script to make cubes
"""
import xml.etree.ElementTree as ET

SIZE = "0.06 0.06 0.06"
COLORS = [
    "1 0.1 0.1 1",
    "1 0.623 0.1 1",
    "0.956 1 0.1 1",
    "0.1 1 0.1 1",
    "0.1 0.1 1 1",
    "0.733 0.101 1 1",
]
NAMES = [
    "red_cube",
    "orange_cube",
    "yellow_cube",
    "green_cube",
    "blue_cube",
    "indigo_cube",
]


for name, color in zip(NAMES,COLORS):

    tree = ET.parse("template_cube.sdf")
    root = tree.getroot()
    for model in root.iter("model"):
        model.set("name", name)
    for diffuse in root.iter("diffuse"):
        diffuse.text = color
    for size in root.iter("size"):
        size.text = SIZE

    tree.write(name + ".sdf")