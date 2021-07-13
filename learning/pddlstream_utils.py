from pddlstream.language.object import Object, OptimisticObject


def logical_to_string(logical):
    """
    Turns an init/goal list for a
    pddl problem into a consistent string
    identifier
    (ie. the order of the predicates will not
    matter for the output string)
    """
    logical = sorted(logical, key=lambda x: str(x))
    res = ""
    num = 0
    for item in logical:
        num += 1
        if isinstance(item, str):
            res += item + ","
            continue
        res += "("
        for i in range(len(item)):
            s = item[i]
            res += str(s)
            if i < len(item) - 1:
                res += ", "
        res += ")"
        if num < len(logical):
            res += ", "
    res += "\n"
    return res


def item_to_dict(atom_map):
    res = {}
    for item in atom_map:
        key = tuple(item[0])
        value = [tuple(i) for i in item[1]]
        res[key] = value
    return res


def ancestors(fact, atom_map):
    """
    Given a fact, return a set of
    that fact's ancestors
    """
    parents = atom_map[fact]
    res = set(parents)
    for parent in parents:
        res |= ancestors(parent, atom_map)
    return set(res)

def siblings(fact, atom_map):
    sib = []
    for can_sib, can_sib_parents in atom_map.items():
        if can_sib_parents == atom_map[fact]:
            sib.append(can_sib)
    return set(sib)

def elders(fact, atom_map):
    """
    Getting uncles and parents
    (certified facts + domain facts
    both count as ancestors) 
    """

    parents = atom_map[fact]
    uncles = set()
    for p in parents:
        uncles |= siblings(p, atom_map)
    res = set(parents) | uncles
    init = res.copy()
    for elder in init:
        res |= elders(elder, atom_map)
    return res

def ancestors_tuple(fact, atom_map):
    """
    Given a fact, return a pre-order from bottom-up tuple of
    that fact's branch
    """
    parents = atom_map[fact]
    ancestors = tuple()
    for parent in parents:
        ancestors += (parent,)
        ancestors += ancestors_tuple(parent, atom_map)

    return ancestors


def make_atom_map(node_from_atom):
    atom_map = {}
    for atom in node_from_atom:
        node = node_from_atom[atom]
        result = node.result
        # TODO: Figure out how to deal with these bools?
        if result is None:
            atom_map[fact_to_pddl(atom)] = []
            continue
        if isinstance(result, bool):
            continue
        atom_map[fact_to_pddl(atom)] = [fact_to_pddl(f) for f in result.domain]
    return atom_map


def fact_to_pddl(fact):
    new_fact = [fact[0]]
    for obj in fact[1:]:
        pddl_obj = obj_to_pddl(obj)
        new_fact.append(pddl_obj)
    return tuple(new_fact)


def obj_to_pddl(obj):
    pddl_obj = (
        obj.pddl
        if isinstance(obj, Object) or isinstance(obj, OptimisticObject)
        else obj
    )
    return pddl_obj


def apply_substitution(fact, substitution):
    return tuple(substitution.get(arg, arg) for arg in fact)


def sub_map_from_init(init):
    objects = objects_from_facts(init)
    return {o: o for o in objects}


def objects_from_facts(init):
    objects = set()
    for fact in init:
        for arg in fact[1:]:
            objects.add(arg)
    return objects



def make_stream_map(node_from_atom):
    stream_map = {}
    for atom in node_from_atom:
        node = node_from_atom[atom]
        result = node.result
        # TODO: Figure out how to deal with these bools?
        if result is None:
            stream_map[fact_to_pddl(atom)] = None
            continue
        if isinstance(result, bool):
            continue
        stream_map[fact_to_pddl(atom)] = result.external.name
    return stream_map

