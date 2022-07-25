
# %%
import argparse
import getpass
import heapq
import itertools
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
from copy import deepcopy
from glob import glob

import numpy as np
import torch
import yaml

USER = getpass.getuser()
sys.path.insert(
    0,
    f"/home/{USER}/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/",
)

from experiments.blocks_world.run_lifted import create_problem
from lifted.a_star import extract_stream_plan_from_path, try_a_star, try_policy_guided_beam
from lifted.a_star import ActionStreamSearch, repeated_a_star, try_policy_guided, stream_cost, _stream_cost
from lifted.utils import PriorityQueue
from pddlstream.language.object import Object
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader


def parse_action(action, pddl_to_name):
    def to_name(op, arg1, arg2):
        return f"{op}(panda,{pddl_to_name[arg1]},{pddl_to_name[arg2]})"

    name, args = action
    if name == 'move':
        return None
    elif name == 'pick':
        block, region = args[1:3]
        return to_name(name, block, region)
    elif name == 'place':
        block, region = args[1:3]
        return to_name(name, block, region)
    elif name == 'stack':
        block, lowerblock = args[1], args[4]
        return to_name(name, block, lowerblock)
    elif name == 'unstack':
        block, lowerblock = args[1], args[4]
        return to_name(name, block, lowerblock)
    raise ValueError(f"Action {name} not recognized")

class State:
    blocks = None
    regions = None

    def copy(self):
        state = State()
        state.blocks = deepcopy(self.blocks)
        state.regions = deepcopy(self.regions)
        return state

    @classmethod
    def from_scene(cls, problem_file):
        SURFACES = {
            'blue_table': {
                'x': [-0.6,  0. ,  0. ],
                'size': [0.4 , 0.75, 0.  ]
            },
            'green_table': {
                'x': [0. , 0.6, 0. ],
                'size': [0.4 , 0.75, 0.  ]
            },
            'purple_table': {
                'x': [ 0. , -0.6,  0. ],
                'size': [0.4 , 0.75, 0.  ]
            },
            'red_table': {
                'x': [0.6, 0. , 0. ],
                'size': [0.4 , 0.75, 0.  ]
            }
        }

        state = State()            
        state.blocks = {}
        state.regions = {}

        for surface in problem_file['surfaces']:
            state.regions[surface] = {
                "x": np.array(SURFACES[surface]['x']),
                "size": np.array(SURFACES[surface]['size']),
                "type": 0
            }

        assert all(predicate in ['on-block', 'on-table'] for predicate, _, _ in problem_file['goal'][1:])
        region_goals = {block: region[0] for predicate, block, region in problem_file['goal'][1:] if predicate == 'on-table'}
        block_goals = {block: lowerblock for predicate, block, lowerblock in problem_file['goal'][1:] if predicate == 'on-block'}
        reverse_block_goal = {lb:b for b,lb in block_goals.items()}
    
        for block in problem_file['objects']:
            
            state.blocks[block] = {
                "type": 1,
                "x": np.array(problem_file['objects'][block]['X_WO'][:3]),
                "x_uncertainty": np.array([0, 0, 0]),
                "held": False,
                "region": problem_file['objects'][block].get('on-table', (None, None))[0],
                "is_below": None,
                "is_above": None,
                "size": np.array([.05, .05, .1 if 'blocker' in block else .05]),
                "goal_region": region_goals.get(block),
                "goal_above": block_goals.get(block),
                "goal_below": reverse_block_goal.get(block)
            }
        for block in problem_file['objects']:
            lowerblock = problem_file['objects'][block].get('on-block')
            if lowerblock is not None:
                state.blocks[lowerblock]['is_below'] = block
                state.blocks[block]['is_above'] = lowerblock
        return state

    def transition(self, action):
        action = parse_action_name(action)
        operator = action[0]
        blocks = self.blocks.copy()
        regions = self.regions
        if operator == 'pick':
            operator, block, region = action
            blocks[block] = blocks[block].copy()
            blocks[block]['held'] = True
            blocks[block]['region'] = None
    
        elif operator == 'place':
            operator, block, region = action
            blocks[block] = blocks[block].copy()
            blocks[block]['held'] = False
            blocks[block]['region'] = region

            blocks[block]['x'] = regions[region]['x']
            blocks[block]['x_uncertainty'] = regions[region]['size'] / 2

        elif operator == 'unstack':
            operator, block, lowerblock = action
            blocks[block] = blocks[block].copy()
            blocks[lowerblock] = blocks[lowerblock].copy()

            blocks[block]['held'] = True
            blocks[lowerblock]['is_below'] = None
            blocks[block]['is_above'] = None
    
        elif operator == 'stack':
            operator, block, lowerblock = action
            blocks[block] = blocks[block].copy()
            blocks[lowerblock] = blocks[lowerblock].copy()

            blocks[block]['is_above'] = lowerblock
            blocks[lowerblock]['is_below'] = block
            blocks[block]['held'] = False

            blocks[block]['x'] = blocks[lowerblock]['x'] + blocks[lowerblock]['size'] * np.array([0,0,1])
            blocks[block]['x_uncertainty'] = blocks[lowerblock]['x_uncertainty']
        else:
            raise ValueError(operator)
        new_state = State()
        new_state.blocks = blocks
        new_state.regions = regions
        return new_state

    @property
    def held_block(self):
        for block in self.blocks:
            if self.blocks[block].get("held"):
                return block
        return None

    def operators(self):
        gripper = 'panda'
        held_block = self.held_block
        if held_block is None:
            for block in self.blocks:
                if self.blocks[block]['is_below'] is not None:
                    continue
                if self.blocks[block]['region'] in self.regions:
                    region = self.blocks[block]['region']
                    yield f"pick({gripper},{block},{region})"
                else:
                    lowerblock = self.blocks[block]['is_above']
                    yield f"unstack({gripper},{block},{lowerblock})"


        else:
            for region in self.regions:
                yield f"place({gripper},{held_block},{region})"
            for block in self.blocks:
                if block == held_block:
                    continue
                if self.blocks[block]["is_below"] is None:
                    yield f"stack({gripper},{held_block},{block})"
            
    def __hash__(self):
        return hash(
            tuple((
                b,
                s["held"],
                tuple(s['x']),
                tuple(s.get('x_uncertainty', 0)),
                s['region'],
                s['is_above'],
                s['is_below']
            ) for b,s in sorted(list(self.blocks.items()))))

def _get_model_input_nodes(state):
    node_feature_to_index = {
        "x": 0,
        "size": 3,
        "held": 6,
        "type": 7,
        # "region": 10,
        # "goal_region": 14
    }
    node_feature_size = 7 + 2
    edge_feature_to_index = {
        "is_above": 0,
        "is_below": 1,
        "goal_above": 2,
        "goal_below": 3,
        "transform": 4
    }
    edge_feature_size = 4 + 3

    nodes = []
    node_info = {}
    for block in sorted(list(state.blocks)):
        nodes.append(block)
        node_info[block] = state.blocks[block]
    for region in sorted(list(state.regions)):
        nodes.append(region)
        node_info[region] = state.regions[region]

    node_features = torch.zeros((len(nodes), node_feature_size))

    for node, node_name in enumerate(nodes):
        block_info = node_info[node_name]
        node_features[node, node_feature_to_index["x"]: node_feature_to_index["x"] + 3] = torch.tensor(block_info["x"], dtype=torch.float)
        node_features[node, node_feature_to_index["held"]:node_feature_to_index["held"] + 1] = torch.tensor(block_info.get("held", 0), dtype=torch.float)
        node_features[node, node_feature_to_index["size"]:node_feature_to_index["size"] + 3] = torch.tensor(block_info["size"], dtype=torch.float)
        node_features[node, node_feature_to_index["type"] + block_info["type"]] = 1
    
        # if block_info["region"] or :
        #     edges.append((node, nodes.index(block_info["region"])))
        #     node_features[node, node_feature_to_index["region"] + region_to_index[block_info["region"]]] = 1
        # if block_info["goal_region"]:
        #     node_features[node, node_feature_to_index["goal_region"] + region_to_index[block_info["goal_region"]]] = 1

    edges = []
    edge_attributes_list = []
    for node, objname in enumerate(nodes):
        for othernode, otherobjname in enumerate(nodes):
            if node == othernode:
                continue
            edge_attributes = torch.zeros((edge_feature_size,))
            obj = node_info[objname]
            otherobj = node_info[otherobjname]
            include_edge = False
            if otherobjname in [obj.get('region'), obj.get("is_above")]:
                edge_attributes[edge_feature_to_index["is_above"]] = 1
                include_edge = True
            if otherobjname == obj.get("is_below") or otherobj.get("region") == objname:
                edge_attributes[edge_feature_to_index["is_below"]] = 1
                include_edge = True
            if otherobjname in [obj.get('goal_region'), obj.get("goal_above")]:
                edge_attributes[edge_feature_to_index["goal_above"]] = 1
                include_edge = True
            if otherobjname == obj.get("goal_below") or otherobj.get('goal_region') == objname:
                edge_attributes[edge_feature_to_index["goal_below"]] = 1
                include_edge = True

            if not include_edge:
                continue

            edge_attributes[edge_feature_to_index["transform"]:edge_feature_to_index["transform"] + 3] = torch.tensor(otherobj["x"] - obj["x"])

            edges.append((node, othernode))
            edge_attributes_list.append(edge_attributes)

    return nodes, node_features, edges, torch.vstack(edge_attributes_list)

def parse_action_name(action_name):
    result = re.search(r'(pick|place|stack|unstack)\(([^,]+),([^,]+),(.+)\)($| )', action_name)
    action, _, block, region = result.group(1), result.group(2), result.group(3), result.group(4)
    try:
        region = eval(region)
    except Exception:
        pass
    if isinstance(region, tuple):
        region = region[0]
    return (action, block, region)

def _get_model_input_actions(nodes, action_name):
    operator, block, region = parse_action_name(action_name)
    action_vocab = {
        "pick": 0,
        "place": 1,
        "stack": 2,
        "unstack": 3
    }
    operator_rep = torch.zeros((1, len(action_vocab)))
    operator_rep[0, action_vocab[operator]] = 1
    target_block_1 = nodes.index(block)
    target_block_2 = nodes.index(region)

    return operator_rep, target_block_1, target_block_2

def get_model_input(state, action_name):
    nodes, node_features, edges, edge_attributes = _get_model_input_nodes(state)
    operator_rep, target_block_1, target_block_2 = _get_model_input_actions(nodes, action_name)

    return nodes, node_features, edges, edge_attributes, operator_rep, target_block_1, target_block_2


def MLP(layers, input_dim):
    mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(torch.nn.ReLU())
        mlp_layers.append(torch.nn.Linear(layers[layer_num],
                                      layers[layer_num + 1]))

    return torch.nn.Sequential(*mlp_layers)


class AttentionPolicy(torch.nn.Module):
    def __init__(self, node_dim=9, edge_dim=7, action_dim=4, N=30, dropout=0, encoder_dim=64, encoder_layers=2, num_heads=4):
        super().__init__()
        self.node_encoder = MLP([encoder_dim] * encoder_layers, node_dim)
        self.edge_encoder = MLP([encoder_dim] * encoder_layers, edge_dim)

        self.encoder_1 = GATv2Conv(in_channels=encoder_dim,
                 out_channels=int(encoder_dim / num_heads), heads = num_heads, edge_dim=encoder_dim, dropout=dropout)
        self.encoder_2 = GATv2Conv(in_channels=encoder_dim,
                 out_channels=int(encoder_dim / num_heads), heads = num_heads, edge_dim=encoder_dim, dropout=dropout)

        self.action_encoder = MLP([encoder_dim] * encoder_layers, action_dim + encoder_dim * 2 + node_dim * 2)
        self.att = GATv2Conv(in_channels=encoder_dim,
                 out_channels=int(encoder_dim / num_heads), heads = num_heads, dropout=dropout)
        self.output = MLP([16, 1], encoder_dim)

    def forward(self, B):
        node_enc = self.node_encoder(B.x)
        edge_attr = self.edge_encoder(B.edge_attr)
        node_enc = self.encoder_1(node_enc, edge_index=B.edge_index, edge_attr=edge_attr)
        node_enc = self.encoder_2(node_enc, edge_index=B.edge_index, edge_attr=edge_attr)

        target_1_enc = node_enc[B.t1_index]
        target_2_enc = node_enc[B.t2_index]
        target_2_enc[B.t2_index == -1] = 0

        target_1_enc_res = B.x[B.t1_index]
        target_2_enc_res = B.x[B.t2_index]
        target_2_enc_res[B.t2_index == -1] = 0

        action_enc = self.action_encoder(torch.cat([B.ops, target_1_enc, target_1_enc_res, target_2_enc, target_2_enc_res], dim=1))
        starting_points = []
        end_points = []
        num_copies_sum = 0
        for index, num_copies in enumerate(B.num_ops):
            num_nodes = B.node_count[index]
            starting_points.append((torch.arange(num_nodes) + B.ptr[index]).repeat(num_copies))
            end_points.append(torch.arange(num_copies).repeat_interleave(num_nodes) + num_copies_sum + B.num_nodes)
            num_copies_sum += num_copies

        attention_edges = torch.cat(
            [
                torch.cat(starting_points, 0).unsqueeze(1),
                torch.cat(end_points, 0).unsqueeze(1)
            ],
            dim=1
        ).t().contiguous()

        attended_actions = self.att(torch.vstack([node_enc, action_enc]), attention_edges)[B.num_nodes:]
        logits = self.output(attended_actions)
        
        return logits

def group(logits, num_ops):
    groups = []
    s = 0
    for i, n in enumerate(num_ops):
        groups.append(logits[s: s + n].squeeze(1).unsqueeze(0))
        s += n
    return groups

def get_data(state):
    nodes, node_features, edges, edge_attributes = _get_model_input_nodes(state)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ops = []
    t1, t2 = [], []
    action_names = []
    for action_name in state.operators():
        operator_rep, target_block_1, target_block_2 = _get_model_input_actions(nodes, action_name)
        ops.append(operator_rep)
        t1.append(target_block_1)
        t2.append(target_block_2)
        action_names.append(action_name)
    ops = torch.vstack(ops)
    t1 = torch.tensor(t1, dtype=torch.long)
    t2 = torch.tensor(t2, dtype=torch.long)
    return Data(
        nodes=nodes,
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attributes,
        ops=ops,
        t1_index=t1,
        t2_index=t2,
        num_ops=len(ops),
        node_count=len(nodes),
        action_names=action_names
    )

def make_dataset(data_files):
    states = []
    all_data = []
    for f in data_files:
        with open(f, 'rb') as fb:
            f_data = pickle.load(fb)
        with open(os.path.join(os.path.dirname(f), 'stats.json'), 'r') as fb:
            stats_data = json.load(fb)
        with open(f"/home/{USER}/drake-tamp/" + stats_data['problem_file_path'].split('drake-tamp/')[1], 'r') as fb:
            problem_file = yaml.safe_load(fb)

        if not stats_data['solution']:
            continue
        # for plan in stats_data['action_plans'][-1:]:
        for plan in [stats_data['solution']]:
            problem_info = f_data['problem_info']
            state = State.from_scene(problem_file)
            pddl_to_name = problem_info.object_mapping
            demonstration = [parse_action(a, pddl_to_name) for a in plan]

            for action in filter(bool, demonstration):
                states.append(state)
                data = get_data(state)
                operator, arg1, arg2 = parse_action_name(action)
                action = f'{operator}(panda,{arg1},{arg2})'
                y = [action in action_name for action_name in data.action_names]
                assert any(y)
                data.y = torch.tensor(y, dtype=torch.long)
                all_data.append(data)
                state = state.transition(action)
    return all_data, states

def make_dataset_lifted(probs):
    states = []
    all_data = []
    for f,d in probs.items():
        f = f.replace('../', '')
        with open(f, 'r') as fb:
            problem_file = yaml.safe_load(fb)

        if not d or not d['solved'] or not "path_data" in d:
            continue

        for plan in [d["path_data"][-1]]:
            state = State.from_scene(problem_file)
            demonstration = [a['action_name'] for a in plan]

            for action in filter(bool, demonstration):
                states.append(state)
                data = get_data(state)
                operator, arg1, arg2 = parse_action_name(action)
                action = f'{operator}(panda,{arg1},{arg2})'
                y = [action in action_name for action_name in data.action_names]
                assert sum(y) == 1
                data.y = torch.tensor(y, dtype=torch.long)
                all_data.append(data)
                state = state.transition(action)
    return all_data, states

def is_goal(state):
    return all((
        (state.blocks[block]['goal_region'] is None or state.blocks[block]['region'] == state.blocks[block]['goal_region']) and \
        (state.blocks[block]['goal_above'] is None or state.blocks[block]['is_above'] == state.blocks[block]['goal_above']) and \
        (state.blocks[block]['goal_below'] is None or state.blocks[block]['is_below'] == state.blocks[block]['goal_below'])) for block in state.blocks)

def levinTs(state, model):
    counter = 0
    queue = [(1, 0, state, 1, [])]
    expansions = 0
    closed = set()
    while queue:
        expansions += 1
        _, _, state, p, history = heapq.heappop(queue)
        if expansions % 100 == 0:
            print(expansions, len(closed), len(history), p)
        
        if is_goal(state):
            print(history)
            print(f'Found goal in {expansions} expansions')
            yield history
            continue
        if hash(state) in closed:
            continue

        for p_a, action in invoke(state, model):
            child = state.transition(action)
            heapq.heappush(queue, ((1*len(history) + 1)/(p_a*p), counter, child, p_a*p, history + [action]))
            counter += 1

        closed.add(hash(state))
def invoke(state, model, temperature=1):
    with torch.no_grad():
        data = get_data(state)
        p_A = torch.nn.functional.softmax(model(Batch.from_data_list([data])).squeeze(1)/temperature).detach().numpy()
    return list(zip(p_A, data.action_names))

def load_model(path='policy.pt'):
    model = AttentionPolicy()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model




#%%
class Policy:
    def __init__(self, problem_file, search, model):
        self.initial_search_state = search.init
        self.initial_state = State.from_scene(problem_file)
        self.planning_states = {
            search.init: self.initial_state
        }
        self.cache = {}
        self.model = model

    def __call__(self, state, op, child):
        if state is self.initial_search_state:
            # the inital state's hash could change as a result of SearchState.make_unique()
            model_state = self.initial_state
        else:
            model_state = self.planning_states[state]
        opname = op.name.split(' ')[0]
        key = (model_state, opname)
        if key not in self.cache:
            for p, op_name in invoke(model_state, self.model):
                op_key = (model_state, op_name)
                self.cache[op_key] = p
        self.planning_states[child] = model_state.transition(opname)
        return self.cache[key]


#%%
class FeedbackPolicy:
    def __init__(self, base_policy, cost_fn, stats, initial_state):
        self._stats = stats
        self._policy = base_policy
        self._cost_fn = cost_fn
        # We use id(state) as a key in these data structures because
        # they store path (not state) dependent quantities.
        self.path_prob = {id(initial_state): 1}
        self.path_length = {id(initial_state): 0}
    
    def __call__(self, state, op, child):
        s = 0
        m_attempts = lambda a: 10**(1 + a // 10)
        for a, c in state.children:
            if self._cost_fn is not None:
                fa = self._cost_fn(state, a, c, stats=self._stats)
            else:
                fa = self._stats.get(a, {"num_successes": 0,"num_attempts": 0})
                
            
            fa = (1 + fa["num_successes"] * m_attempts(fa["num_attempts"])) / (1 +  fa["num_attempts"]*m_attempts(fa["num_attempts"]))

            pa = self._policy(state, a, c)
            s += fa * pa
        
        
        if self._cost_fn is not None:
            fa = self._cost_fn(state, op, child, stats=self._stats)
        else:
            fa = self._stats.get(op, {"num_successes": 0,"num_attempts": 0})
        fa = (1 + fa["num_successes"]*m_attempts(fa["num_attempts"])) / (1 +  fa["num_attempts"]*m_attempts(fa["num_attempts"]))
        pa = self._policy(state, op, child) 
        p_adj = (fa * pa) / s

        self.path_prob[id(child)] = p_adj * self.path_prob[id(state)]
        self.path_length[id(child)] = 1 + self.path_length[id(state)]
        return (1) / self.path_prob[id(child)]

def extract_labels(path, stats):
    stream_plan = extract_stream_plan_from_path(path)
    (_,_,initial_state), _ = stream_plan[0]


    path_data = []
    for (edge, object_map) in stream_plan:
        action = edge[1]
        action_feasibility = stats.get(action)
        if not action_feasibility:
            break

        action_streams = list({action for action in object_map.values()})
        stream_feasibility = [stats[edge[2].id_anon_cg_map[stream.outputs[0]]] for stream in action_streams]
        path_data.append(dict(
            action_name=action.name,
            action_feasibility=action_feasibility,
            streams=[stream.serialize() for stream in action_streams],
            stream_feasibility=stream_feasibility
        ))
    return path_data        

def train_model(model, train_dataset, valid_dataset, model_path, epochs):

    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=128)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    def loss(logits, targets, num_ops):
        count = 0
        total = 0
        s = 0
        for i, n in enumerate(num_ops):
            total += criterion(logits[s: s + n].squeeze(1).unsqueeze(0), targets[s:s + n].nonzero().squeeze(0))
            s += n
            count += 1
        return total/count

    for i in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            pred = model(batch)
            l = loss(pred, batch.y, batch.num_ops)
            train_loss += l.item()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = train_loss / len(train_loader)
        val_loss = 0
        model.eval()
        for batch in valid_loader:
            pred = model(batch)
            l = loss(pred, batch.y, batch.num_ops)
            val_loss += l.item()
        if len(valid_loader):
            val_loss = val_loss / len(valid_loader)
        print(i, train_loss, val_loss)

    torch.save(model.state_dict(), model_path)

def make_policy(problem_file, search, model, stats):
    base_policy = Policy(problem_file, search, model)
    policy = FeedbackPolicy(base_policy=base_policy, cost_fn=_stream_cost, stats=stats, initial_state=search.init)
    return policy

def run_problem(problem_file_path, model):
    with open(problem_file_path, 'r') as f:
        problem_file = yaml.full_load(f)

    init, goal, externals, actions = create_problem(problem_file_path)

    if len(problem_file['objects']) == 1:
        return # policy expects edges yo!

    search = ActionStreamSearch(init, goal, externals, actions)
    if search.test_goal(search.init):
        return
    stats = {}
    policy = make_policy(problem_file, search, model, stats)
    r = repeated_a_star(search, search_func=try_policy_guided, stats=stats, policy_ts=policy, max_steps=100, edge_eval_steps=10, max_time=90, debug=False)
    path_data = []
    for path in r.skeletons:
        path_data.append(extract_labels(path, r.stats))
    objects = ({o_pddl:dict(name=o_pddl,value=Object.from_name(o_pddl).value) for o_pddl in search.init_objects})
    return dict(
        name=os.path.basename(problem_file_path),
        planning_time=r.planning_time,
        solved=r.solution is not None,
        goal=goal,
        objects=objects,
        path_data=path_data,
        expanded=r.expand_count,
        evaluated=r.evaluate_count,
    )

def split_data(data, portion=.9):
    keys = list(data.keys())
    np.random.shuffle(keys)
    total = len(keys)
    split = int(total * portion)
    train = keys[:split]
    valid = keys[split:]
    return {k:data[k] for k in train}, {k:data[k] for k in valid}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dataset_json_path", type=str, default=None)
    parser.add_argument("--dataset_pkl_paths", type=str, default="data/leapfrog_data_clutter.pkl")
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=500)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.dataset_json_path:
        with open(args.dataset_json_path, "r") as f:
            all_files = json.load(f)
        train_dset, _ = make_dataset(all_files["train"])
        val_dset, _ = make_dataset(all_files["validation"])
    elif args.dataset_pkl_paths:
        pkl_paths = args.dataset_pkl_paths.split(',')
        data = {}
        for pkl_path in pkl_paths:
            with open(pkl_path, 'rb') as f:
                data.update(pickle.load(f))
        train_data, val_data = split_data(data, .85)
        print(len(train_data), len(val_data))
        train_dset, _ = make_dataset_lifted(train_data)
        val_dset, _ = make_dataset_lifted(val_data)
    else:
        raise ValueError("Did not pass a datset option.")
    print('Training model')
    model = AttentionPolicy(dropout=args.dropout)
    if args.start_from:
        model.load_state_dict(torch.load(args.start_from))
    train_model(model, train_dset, val_dset, args.model_path, args.epochs)
