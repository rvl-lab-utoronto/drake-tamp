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

    sib = set()
    for can_sib, can_sib_parents in atom_map.items():
        if can_sib_parents == atom_map[fact]:
            sib.add(can_sib)

    return sib

def make_sibling_map(atom_map):

    res = {}
    for sib, parents in atom_map.items():
        res.setdefault(tuple(parents), set()).add(sib)
    return res

def get_siblings_from_map(fact, atom_map, sibling_map):
    chl = sibling_map[tuple(atom_map[fact])]
    sib = chl.copy()
    sib.remove(fact)
    return sib

def elders(fact, atom_map, level, sibling_map, elders_cache = {}):
    """
    Getting uncles and parents
    (certified facts + domain facts
    both count as ancestors) 

    level is a dictionary mapping from fact to its level
    """

    parents = atom_map[fact]
    uncles = set()
    for p in parents:
        uncles |= get_siblings_from_map(p, atom_map, sibling_map)#siblings(p, atom_map)
    res = set(parents) | uncles
    init = res.copy()
    for elder in init:
        if elder in elders_cache:
            res |= elders_cache[elder]
        else:
            res |= elders(elder, atom_map, level, sibling_map, elders_cache = elders_cache)

    elders_cache[fact] = res

    if len(parents) == 0:
        level[fact] = 0
    else:
        level[fact] = 1 + max([level[e] for e in init])

    return res

# depreciated
def dep_elders(fact, atom_map):
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
        res |= dep_elders(elder, atom_map)
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

def standardize_facts(facts, init_objects):
    seen = {o:o for o in init_objects}
    new_facts = tuple()
    for fact in facts:
        new_fact = (fact[0], )
        for arg in fact[1:]:
            if arg not in seen:
                seen[arg] = str(len(seen))
            new_fact = new_fact + (seen[arg],)
        new_facts = new_facts + (new_fact, )

    return new_facts

def objects_from_fact(fact):
    return {arg for arg in fact[1:]}

def objects_from_fact(fact):

    obj_to_inds = dict()
    res = set()
    for i, arg in enumerate(fact[1:]):
        ind = i + 1
        obj_to_inds[arg] = ind
        res.add(arg)
    return res, obj_to_inds

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

