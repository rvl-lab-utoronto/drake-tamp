

from integrate_planner import instantiate_planning_problem, Atom, externals
from lifted_search import ActionStreamSearch, repeated_a_star

world = {
    "width": 20,
    "height": 10
}
regions = {
    "r1": {"width": 10, "x": 0, "y": -0.3, "height": .3},
    "r2": {"width": 4, "x": 16, "y": -0.3, "height": .3},
}

grippers = {
    "g1": {"width": 5, "height": 1, "x": 2, "y": 8, "color": None}
}

blocks = {
    "b0": {
        "x": 0,
        "width": 2,
        "y": 0,
        "height": 1,
        "color": 'blue'
    }
}

objects, actions, initial_state = instantiate_planning_problem((world, grippers, regions, blocks))
goal = set()
goal.add(Atom('on', (objects['b0'].pddl, objects['r2'].pddl)))
search = ActionStreamSearch(initial_state, goal, externals, actions)
stats = {}
result = repeated_a_star(search, stats=stats, max_steps=50)