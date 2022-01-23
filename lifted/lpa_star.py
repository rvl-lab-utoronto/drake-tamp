from datetime import datetime


def try_lpa_star(search, max_step=100000):
    def cost(state, op, child, verbose=False):
        included = set()
        c = 0
        for obj in child.object_stream_map:
            stream_action = child.object_stream_map[obj]
            if stream_action is not None:
                if stream_action in included:
                    continue
                cg_key = child.get_object_computation_graph_key(obj)
                if cg_key in stats:
                    s = stats[cg_key]
                    comp_cost = (
                        (s["num_successes"] + 1) / (s["num_attempts"] + 1)
                    ) ** -1
                    c += comp_cost

                else:
                    comp_cost = 1
                    c += comp_cost
                if verbose:
                    print(stream_action, comp_cost)
                included.add(stream_action)
        return max(1, c)

    def heuristic(state):
        actions = [a for _, a, _ in state.get_shortest_path_to_start()]
        if len(actions) >= 2:
            if actions[-1] is not None and "move" in actions[-1].name:
                if actions[-2] is not None and "move" in actions[-2].name:
                    return np.inf
        return len(search.goal - state.state) * 4
        return 0

    start_time = datetime.now()
    search.init.rhs = 0
    search.start_distance = np.inf
    expand_count = 0
    evaluate_count = 0

    def compute_key(node):
        return (
            min(node.start_distance, node.rhs) + heuristic(node),
            min(node.start_distance, node.rhs),
        )

    def update_node(node):
        if node != search.init:
            node.rhs = min(
                [
                    pred.start_distance + cost(pred, op, node)
                    for op, pred in node.parents
                ]
            )
        if node in q:
            q.remove(node)
        if node.start_distance != node.rhs:
            q.push(node, compute_key(node))

    def shortest_path():
        while (
            q.top_key() < compute_key(search.init)
            or search.goal.rhs != search.goal.start_distance
        ):

            state = q.pop()
            expand_count += 1

            if not state.expanded and not search.goal_test(state):
                for op, child in search.successor(state):
                    evaluate_count += 1
                    if search.goal_test(child):
                        child = search.goal
                        child.rhs = np.inf
                        child.start_distance = np.inf

            if state.start_distance > state.rhs:
                state.start_distance = state.rhs
                for op, child in search.successor(state):
                    update_node(child)
            else:
                state.start_distance = np.inf
                update_node(state)
                for op, child in search.successors(state):
                    update_node(child)

    q = PriorityQueue()
    q.push(search.init, compute_key(search.init))

    while True:
        shortest_path()

        if goal_state is None:
            print("Could not find feasable action plan!")
            break

        print("Getting path ...")
        path = goal_state.get_shortest_path_to_start()
        c = 0
        for idx, i in enumerate(path):
            print(idx, i[1])
            a = cost(*i, verbose=True)
            print("action cost:", a)
            print("cum cost:", a + c)
            c += a

        action_skeleton = [a for _, a, _ in goal_state.get_shortest_path_to_start()]
        actions_str = "\n".join([str(a) for a in action_skeleton])
        print(f"Action Skeleton:\n{actions_str}")

        stream_plan = extract_stream_plan_from_path(path)
        stream_ordering = extract_stream_ordering(stream_plan)
        if not stream_ordering:
            break
        object_mapping = sample_depth_first_with_costs(
            stream_ordering, goal_state, stats
        )
        if object_mapping is not None:
            break
        print("Could not find object_mapping, retrying with updated costs")

        for parent, op, child in edges:
            update_node(child)

    av_branching_f = evaluate_count / expand_count
    approx_depth = math.log(evaluate_count) / math.log(av_branching_f)
    print(f"Explored {expand_count}. Evaluated {evaluate_count}")
    print(f"Av. Branching Factor {av_branching_f:.2f}. Approx Depth {approx_depth:.2f}")
    print(f"Time taken: {(datetime.now() - start_time).seconds} seconds")
