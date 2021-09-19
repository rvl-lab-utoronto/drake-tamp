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


def make_hand_and_block(name, color):
    tree = ET.parse(f"{FILE_PATH}/{TEMPLATE_NAME}")
    root = tree.getroot()
    for model in root.iter("model"):
        model.set("name", name)
    for diffuse in root.iter("diffuse"):
        diffuse.text = color

    return tree


if __name__ == "__main__":
    name = "test_hand_and_block"
    tree = make_hand_and_block(name, '0 1 0 1')
    tree.write(name + ".sdf")
