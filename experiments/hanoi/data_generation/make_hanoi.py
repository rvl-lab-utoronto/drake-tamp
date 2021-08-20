import numpy as np
import yaml
import os
import xml.etree.ElementTree as ET

RED_PEG = np.array([0.755, 0, 0.74])
BLUE_PEG = np.array([0.605, -0.15, 0.74])
GREEN_PEG = np.array([0.605, 0.15, 0.74])

PEGS = {
    "red_peg": RED_PEG,
    "blue_peg": BLUE_PEG,
    "green_peg": GREEN_PEG
}

LENGTH = 0.02
DIRECTIVE = os.path.expanduser(
    "~/drake-tamp/panda_station/directives/hanoi.yaml"
)

def make_disc(name, color, size, ball_radius = 1e-7):
    template_path = os.path.expanduser(
        "~/drake-tamp/panda_station/models/hanoi/sdf/template_disc.sdf"
    )
    tree = ET.parse(os.path.join(template_path))
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
        n = elm.attrib.get("name", "")
        if n.startswith("ball"):
            num = int(n[-1]) - 1
            x = (-1) ** (num & 0b01) * r/np.sqrt(2)
            y = (-1) ** ((num & 0b10) >> 1) * r/np.sqrt(2)
            elm.find("pose").text = f"{x} {y} {0} 0 0 0"
            elm.find("geometry").find("sphere").find("radius").text = f"{ball_radius}"
        elif n== "visual":
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("cylinder").find("radius").text = f"{r}"
            elm.find("geometry").find("cylinder").find("length").text = f"{l}"
        elif n== "collision":
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("cylinder").find("radius").text = f"{r}"
            elm.find("geometry").find("cylinder").find("length").text = f"{l}"
    tree.write(
        os.path.expanduser(
            f"~/drake-tamp/panda_station/models/hanoi/sdf/{name}.sdf"
        )
    )
    return tree

def make_hanoi_problem(num_discs, start_peg, end_peg):

    assert num_discs <= 12
    assert start_peg in PEGS.keys()
    assert end_peg in PEGS.keys()

    yaml_data = {
        "directive": "directives/hanoi.yaml",
        "planning_directive": "directives/hanoi.yaml",
        "arms": {
            "panda": {
                "panda_name": "panda",
                "hand_name": "hand",
                "X_WB": [0.05, 0, 0.8, 0, 0, 0],
            }
        },
        "objects": {},
        "main_links": {
            "wooden_table": "base_link",
        },
        "surfaces": {
            "wooden_table": ["red_peg", "green_peg", "blue_peg"],
        },
    }

    prev_disc = None
    discs = []
    for i in range(num_discs):
        r_int = 32 - 2*i #0.030 - 0.001 * i
        h = i*LENGTH
        name = f"disc_{r_int}"
        r = float(r_int)/1000.0
        if not os.path.isfile(os.path.expanduser(f"~/drake-tamp/panda_station/models/hanoi/sdf/{name}.sdf")):
            red,green,blue = np.random.uniform(0, 1, size = (3,))
            make_disc(name,  f"{red} {green} {blue} 1", np.array([r, LENGTH]))
        pos = PEGS[start_peg]
        pose = np.concatenate((pos, np.zeros(3)))
        pose[2] += h
        if len(discs) == 0:
            yaml_data["objects"][name] = {
                "path": f"models/hanoi/sdf/{name}.sdf",
                "X_WO": pose.tolist(),
                "main_link": "base_link",
                "on-peg": ["wooden_table", start_peg],
                "radius": r_int
            }
        else:
            yaml_data["objects"][name] = {
                "path": f"models/hanoi/sdf/{name}.sdf",
                "X_WO": pose.tolist(),
                "main_link": "base_link",
                "on-disc": discs[-1],
                "radius": r_int
            }
        discs.append(name)

    goal = [
        "and",
        ["on-peg", discs[0], ["wooden_table", end_peg]]
    ]

    for top,bottom in zip(discs[1:], discs[:-1]):
        goal.append(["on-disc", top, bottom])

    yaml_data["goal"] = goal

    yaml_data["run_attr"] = {
        "num_discs": num_discs,
        "start_peg": start_peg,
        "end_peg": end_peg
    }

    return yaml_data


if __name__ == "__main__":

    yaml_data = make_hanoi_problem(4, "blue_peg", "green_peg")
    with open("test_problem.yaml", "w") as stream:
        yaml.dump(yaml_data, stream, default_flow_style=False)