# cat /tmp/stats.json | jq -r .pddl_problems[0] > /tmp/problem.pddl
# cat /tmp/problem.pddl
./pddlstream/FastDownward/fast-downward.py  experiments/gripper2d/domain-manual.pddl /tmp/problem.pddl --heuristic "h=blind()" --search "astar(h,cost_type=PLUSONE)"