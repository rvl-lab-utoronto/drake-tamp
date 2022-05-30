
# %%
from glob import glob
import json
import pickle
import os
import yaml
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
import re
import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
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
            }

        assert all(predicate in ['on-block', 'on-table'] for predicate, _, _ in problem_file['goal'][1:])
        region_goals = {block: region[0] for predicate, block, region in problem_file['goal'][1:] if predicate == 'on-table'}
        block_goals = {block: lowerblock for predicate, block, lowerblock in problem_file['goal'][1:] if predicate == 'on-block'}
        reverse_block_goal = {lb:b for b,lb in block_goals.items()}
    
        for block in problem_file['objects']:
            
            state.blocks[block] = {
                "x": np.array(problem_file['objects'][block]['X_WO'][:3]),
                "x_uncertainty": np.array([0, 0, 0]),
                "held": False,
                "region": problem_file['objects'][block].get('on-table', (None, None))[0],
                "is_below": None,
                "is_above": None,
                "size": np.array([.5, .5, 1. if 'blocker' in block else .5]),
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
        new_state = self.copy()
        if operator == 'pick':
            operator, block, region = action
            new_state.blocks[block]['held'] = True
            new_state.blocks[block]['region'] = None
    
        elif operator == 'place':
            operator, block, region = action
            new_state.blocks[block]['held'] = False
            new_state.blocks[block]['region'] = region

            new_state.blocks[block]['x'] = new_state.regions[region]['x']
            new_state.blocks[block]['x_uncertainty'] = new_state.regions[region]['size'] / 2

        elif operator == 'unstack':
            operator, block, lowerblock = action
            new_state.blocks[block]['held'] = True
            new_state.blocks[lowerblock]['is_below'] = None
            new_state.blocks[block]['is_above'] = None
    
        elif operator == 'stack':
            operator, block, lowerblock = action
            new_state.blocks[block]['is_above'] = lowerblock
            new_state.blocks[lowerblock]['is_below'] = block
            new_state.blocks[block]['held'] = False

            new_state.blocks[block]['x'] = new_state.blocks[lowerblock]['x'] + new_state.blocks[lowerblock]['size'] * np.array([0,0,1])
            new_state.blocks[block]['x_uncertainty'] = new_state.blocks[lowerblock]['x_uncertainty']
        else:
            raise ValueError(operator)
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
        "x_uncertainty": 3,
        "size": 6,
        "held": 9,
        "region": 10,
        "goal_region": 14
    }
    edge_feature_to_index = {
        "is_above": 0,
        "is_below": 1,
        "goal_above": 2,
        "goal_below": 3,
        "transform": 4
    }
    regions = state.regions
    blocks = state.blocks
    nodes = sorted(list(blocks))
    node_to_block = dict(enumerate(nodes))
    node_features = torch.zeros((len(blocks), 18))
    region_to_index = {r:i for i, r in enumerate(regions)}

    for node in node_to_block:
        block = node_to_block[node]
        block_info = blocks[block]
        node_features[node, node_feature_to_index["x"]: node_feature_to_index["x"] + 3] = torch.tensor(block_info["x"], dtype=torch.float)
        node_features[node, node_feature_to_index["x_uncertainty"]:node_feature_to_index["x_uncertainty"] + 3] = torch.tensor(block_info["x_uncertainty"], dtype=torch.float)
        node_features[node, node_feature_to_index["held"]:node_feature_to_index["held"] + 1] = torch.tensor(block_info["held"], dtype=torch.float)
        node_features[node, node_feature_to_index["size"]:node_feature_to_index["size"] + 3] = torch.tensor(block_info["size"], dtype=torch.float)
        if block_info["region"]:
            node_features[node, node_feature_to_index["region"] + region_to_index[block_info["region"]]] = 1
        if block_info["goal_region"]:
            node_features[node, node_feature_to_index["goal_region"] + region_to_index[block_info["goal_region"]]] = 1

    edges = []
    edge_attributes = torch.zeros((len(blocks)**2 - len(blocks), 7))
    for node in node_to_block:
        for othernode in node_to_block:
            if node == othernode:
                continue
            block = node_to_block[node]
            otherblock = node_to_block[othernode]
            block_info = blocks[block]
            edge_attributes[len(edges), edge_feature_to_index["is_above"]] = 1 if otherblock == block_info["is_above"] else 0
            edge_attributes[len(edges), edge_feature_to_index["is_below"]] = 1 if otherblock == block_info["is_below"] else 0
            edge_attributes[len(edges), edge_feature_to_index["goal_above"]] = 1 if otherblock == block_info["goal_above"] else 0
            edge_attributes[len(edges), edge_feature_to_index["goal_below"]] = 1 if otherblock == block_info["goal_below"] else 0
            edge_attributes[len(edges), edge_feature_to_index["transform"]:edge_feature_to_index["transform"] + 3] = torch.tensor(blocks[otherblock]["x"] - block_info["x"])
            edges.append((node, othernode))

    return nodes, node_features, edges, edge_attributes

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
        "pick.green_table": 0,
        "pick.red_table": 1,
        "pick.blue_table": 2,
        "pick.purple_table": 3,
        "place.green_table": 4,
        "place.red_table": 5,
        "place.blue_table": 6,
        "place.purple_table": 7,
        "stack": 8,
        "unstack": 9
    }
    operator_rep = torch.zeros((1, len(action_vocab)))
    operator_rep[0, action_vocab[f"{operator}.{region}" if operator in ["pick", "place"] else operator]] = 1
    target_block_1 = nodes.index(block)
    target_block_2 = -1 if operator in ["pick", "place"] else nodes.index(region)

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
    # if len(layers) > 1:
    #     mlp_layers.append(torch.nn.LayerNorm(
    #         mlp_layers[-1].weight.size()[:-1]))  # type: ignore

    return torch.nn.Sequential(*mlp_layers)


class AttentionPolicy(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, action_dim, N=30):
        super().__init__()
        node_encoder_dim = 32
        self.node_encoder = MLP([16, node_encoder_dim], node_dim)

        self.encoder = GATv2Conv(in_channels=node_encoder_dim,
                 out_channels=int(node_encoder_dim / 2), heads = 2, edge_dim=edge_dim)

        num_heads = 4
        action_embed_dim = 32
        self.action_encoder = MLP([16, action_embed_dim], action_dim + node_encoder_dim * 2)
        self.att = GATv2Conv(in_channels=node_encoder_dim,
                 out_channels=int(node_encoder_dim / 2), heads = 2)
        self.output = MLP([16, 1], action_embed_dim)

    def forward(self, B):
        node_enc = self.node_encoder(B.x)
        node_enc = self.encoder(node_enc, edge_index=B.edge_index, edge_attr=B.edge_attr)

        target_1_enc = node_enc[B.t1_index]
        target_2_enc = node_enc[B.t2_index]

        action_enc = self.action_encoder(torch.cat([B.ops, target_1_enc, target_2_enc], dim=1))
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
        f = f.replace('/jobs/', '/home/mohammed/drake-tamp/data/jobs/')
        with open(f, 'rb') as fb:
            f_data = pickle.load(fb)
        with open(os.path.join(os.path.dirname(f), 'stats.json'), 'r') as fb:
            stats_data = json.load(fb)
        with open('/home/mohammed/drake-tamp/' + stats_data['problem_file_path'].split('drake-tamp/')[1], 'r') as fb:
            problem_file = yaml.full_load(fb)

        if not stats_data['solution']:
            continue
        if 'non_monotonic' in stats_data['problem_file_path']:
            continue
        # for plan in stats_data['action_plans'][-1:]:
        for plan in [stats_data['solution']]:
            # plan = stats_data['solution']
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

#%%
def is_goal(state):
    return all((
        (state.blocks[block]['goal_region'] is None or state.blocks[block]['region'] == state.blocks[block]['goal_region']) and \
        (state.blocks[block]['goal_above'] is None or state.blocks[block]['is_above'] == state.blocks[block]['goal_above']) and \
        (state.blocks[block]['goal_below'] is None or state.blocks[block]['is_below'] == state.blocks[block]['goal_below'])) for block in state.blocks)

import heapq
def levinTs(state, model):
    counter = 0
    queue = [(1, 0, state, 1, [])]
    expansions = 0
    closed = set()
    while queue:
        expansions += 1
        if expansions % 100 == 0:
            print(expansions, len(closed))
        _, _, state, p, history = heapq.heappop(queue)
        
        if is_goal(state):
            print(history)
            print(f'Found goal in {expansions} expansions')
            yield state
            continue
        if hash(state) in closed:
            continue

        for p_a, action in invoke(state, model):
            child = state.transition(action)
            heapq.heappush(queue, ((1*len(history) + 1)/(p_a*p), counter, child, p_a*p, history + [action]))
            counter += 1

        closed.add(hash(state))
def invoke(state, model):
    data = get_data(state)
    p_A = torch.nn.functional.softmax(model(Batch.from_data_list([data])).squeeze(1)).detach().numpy()
    return list(zip(p_A, data.action_names))
#%%
if __name__ == '__main__':

    with open('/home/mohammed/drake-tamp/data/jobs/blocksworld-dset.json', 'r') as f:
        data_files = json.load(f)["train"]

    all_data, _ = make_dataset(data_files)



#%%
    train_loader = DataLoader(all_data, shuffle=True, batch_size=128)
    model = AttentionPolicy(18,7,10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

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

    for i in range(100):
        for batch in train_loader:
            pred = model(batch)
            l = loss(pred, batch.y, batch.num_ops)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(l)



# %%
