"""This experiment looks at a very simple domain where scenarios only differ by the placement of a fixed number (2) of blocks which
have fixed sizes. We want to see if it is possible to use supervised learning to predict the feasibility of picking up each of the
objects, given their placements."""
#%%
import matplotlib.pyplot as plt
import numpy as np

## PDDLSTREAM STUFF
from pddlstream.language.stream import Stream, StreamInfo
from pddlstream.language.object import Object, OptimisticObject
from pddlstream.language.generator import from_gen, from_gen_fn, from_test
from experiments.gripper2d.problem import grasp, placement, ik, check_safe, generate_scene

# make streams
stream_info = StreamInfo(use_unique=True)
grasp_stream = Stream('grasp', from_gen_fn(grasp), ('?gripper', '?block'), [],('?grasp',), [], info=stream_info)
placement_stream = Stream('placement', from_gen_fn(placement), ('?block', '?region'), [],('?place',), [], info=stream_info)
ik_stream = Stream('ik', from_gen_fn(lambda g,bp,gr: ik(g, bp, gr, dict(width=10, height=10))), ('?gripper', '?block_pose', '?handpose'), [],('?conf',), [], info=stream_info)
safety_stream = Stream('check_safe', from_test(check_safe), ('?gripper', '?gripperpose', '?block', '?blockpose'), [],tuple(), [], info=stream_info)


def scene_objects(world, grippers, regions, blocks):
    Object.reset()
    objects = {}
    for region_name in regions:
        objects[region_name] = Object(regions[region_name])
    for gripper_name in grippers:
        objects[gripper_name] = Object(grippers[gripper_name])
        objects[f"{gripper_name}_pose"] = Object([grippers[gripper_name]['x'], grippers[gripper_name]['y']])
    for block_name in blocks:
        objects[block_name] = Object(blocks[block_name])
        objects[f"{block_name}_pose"] = Object([blocks[block_name]['x'], blocks[block_name]['y']])
    return objects

## SAMPLING STUFF

from collections import namedtuple
from lifted_search import get_stream_action_edges, Binding, StreamAction
DummyStream = namedtuple('DummyStream', ['name'])
def ancestral_sampling(objects, stream_ordering):
    objects_from_name = {v.pddl:v for v in objects.values()}
    nodes = stream_ordering
    edges = get_stream_action_edges(stream_ordering)
    final_node = StreamAction(
        DummyStream('FINAL'),
        inputs=tuple(obj for stream_action in nodes for obj in stream_action.outputs),
        outputs=tuple()
    )
    start_node = StreamAction(
        DummyStream('START'),
        inputs=tuple(),
        outputs=tuple()
    )
    children = {
    }
    for parent, child in edges:
        children.setdefault(parent, set()).add(child)
    for node in nodes:
        children.setdefault(node, set()).add(final_node)
        children.setdefault(start_node, set()).add(node)
    stats = {
        node: 0
        for node in nodes
    }
    produced = dict()
    queue = [Binding(0, [start_node], {})]
    while queue:
        binding = queue.pop(0)
        stream_action = binding.stream_plan[0]
        if stream_action not in [start_node, final_node]:   
            input_objects = [produced.get(var_name) or objects_from_name[var_name] for var_name in stream_action.inputs]
            fluent_facts = [(f.predicate, ) + tuple(produced.get(var_name) or objects_from_name[var_name] for var_name in f.args) for f in stream_action.fluent_facts]
            stream_instance = stream_action.stream.get_instance(input_objects, fluent_facts=fluent_facts)
            if stream_instance.enumerated:
                continue
            results, new_facts = stream_instance.next_results(verbose=False)
            if not results:
                continue
            [new_stream_result] = results
            output_objects = new_stream_result.output_objects
            if stream_action.stream.is_test:
                output_objects = (True,)
            newly_produced = dict(zip(stream_action.outputs, output_objects))
            for obj in newly_produced:
                produced[obj] = newly_produced[obj]
            # new_mapping = binding.mapping.copy()
            # new_mapping.update(newly_produced)
    
        else:
            # new_mapping = binding.mapping
            pass

        stats[stream_action] = stats.get(stream_action, 0) + 1
        for child in children.get(stream_action, []):
            input_objects = list(child.inputs) + [var_name for f in child.fluent_facts for var_name in f.args]
            if all(obj in produced or obj in objects_from_name for obj in input_objects):
                new_binding = Binding(binding.index + 1, [child], {})
                queue.append(new_binding)
    return stats

def ancestral_sampling_acc(objects, ordering, num_attempts=10):
    stats = {}
    for i in range(num_attempts):
        s = ancestral_sampling(objects, ordering)
        for n in s:
            if n.stream.name is 'START' or n.stream.name is 'FINAL':
                stats[n.stream.name] = stats.get(n.stream.name, 0) + s[n]
            else:
                stats[n] = stats.get(n, 0) + s[n]
    return stats

def pick_block_cg(objects, block_name):
    return [
        StreamAction(grasp_stream, (objects['g1'].pddl, objects[block_name].pddl), ('?grasp',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[block_name + '_pose'].pddl, '?grasp'), ('?conf',)),
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf', objects['b0'].pddl, objects['b0_pose'].pddl), ('?safe1',)),
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf', objects['b1'].pddl, objects['b1_pose'].pddl), ('?safe2',)),
    ]

## MODEL STUFF
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_fixed_size_vec(scene):
    blocks = scene[-1]
    rep = [
        blocks['b0']['x'],
#         blocks['b0']['y'],
#         blocks['b0']['width'],
#         blocks['b0']['height'],
        blocks['b1']['x'],
#         blocks['b1']['y'],
#         blocks['b1']['width'],
#         blocks['b1']['height']
    ]
    return rep

def dataset_to_XY(dataset):
    X = []
    Y = []
    for i in range(0, len(dataset), 2):
        (scene, stats1) = dataset[i]
        (_, stats2) = dataset[i + 1]
        assert _ is scene
        X.append(get_fixed_size_vec(scene))
        Y.append([
            stats1.get('FINAL', 0) / stats1['START'],
            stats2.get('FINAL', 0) / stats2['START']
        ])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train(model, optimizer, criterion, X, Y, epochs=1000, batch_size=10):
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)

    dataset = TensorDataset(X,Y) # create your datset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # create your dataloader

    model.train()
    for e in range(epochs):
        for xb, yb in dataloader:
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def eval(model, X_test, Y_test, log=False):
    model.eval()

    preds = model(torch.Tensor(X_test))
    preds = preds.detach().numpy()
    if log:
        preds = np.exp(preds)
    l1_loss = np.abs(preds - Y_test)
    # print(f"\t avg l1: {l1_loss.mean(0)} {l1_loss.mean()}")

    pred_easier = np.argmax(preds, 1)
    true_easier = np.argmax(Y_test, 1)
    acc = (pred_easier == true_easier).mean()
    # print(f"\t acc: {acc}")
    return l1_loss.mean(), acc
def generate_dataset(num_scenarios=1000, num_ancestral_samples=30):
    dataset = []
    for i in range(num_scenarios):
        scene = generate_scene([1, 3])
        objects = scene_objects(*scene)
        

        ordering = pick_block_cg(objects, 'b0')
        stats = ancestral_sampling_acc(objects, ordering, num_ancestral_samples)
        dataset.append((scene, stats))

        ordering = pick_block_cg(objects, 'b1')
        stats = ancestral_sampling_acc(objects, ordering, num_ancestral_samples)
        dataset.append((scene, stats))
    return dataset

def MLP(layers, input_dim, log=False):
    mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(torch.nn.LeakyReLU())
        mlp_layers.append(torch.nn.Linear(layers[layer_num], layers[layer_num + 1]))
    if log:
        mlp_layers.append(torch.nn.LogSigmoid())
    else:
        mlp_layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*mlp_layers)


def eval_effect_of_training_size(test_size = 1000, train_sizes=[10, 50, 100, 500, 1000], num_trials=5):
    test_dataset = generate_dataset(test_size)
    X_test, Y_test = dataset_to_XY(test_dataset)

    train_dataset = generate_dataset(2 * max(train_sizes))
    X_train, Y_train = dataset_to_XY(train_dataset)

    results = {}
    for size in train_sizes:
        for trial in range(num_trials):
            model = MLP([16, 2], 2)
            criterion = torch.nn.L1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            idxs = np.random.choice(len(X_train), size=size, replace=False)
            X, Y = X_train[idxs, :], Y_train[idxs, :]
            train(model, optimizer, criterion, X, Y, batch_size=min(size, 100), epochs=50)
            loss, acc = eval(model, X_test, Y_test)
            results.setdefault(size, []).append({"l1": loss, "acc": acc})
    
    y = []
    yerr = []
    for size in train_sizes:
        acc = []
        for r in results[size]:
            acc.append(r['acc'])
        y.append(np.mean(acc))
        yerr.append(np.std(acc))
    plt.errorbar(train_sizes, y, yerr=yerr)

    x = []
    y = []
    for size in train_sizes:
        for r in results[size]:
            x.append(size)
            y.append(r['acc'] + (np.random.random() - .5)*1e-2)
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel('Training Size')
    plt.ylabel('Test Accuracy')

    return results

def eval_sampling_variance_ancestral(num_scenarios=1000, num_ancestral_samples=30, num_trials=5):
    b1 = []
    b2 = []
    for i in range(num_scenarios):
        scene = generate_scene([1, 3])
        objects = scene_objects(*scene)
        
        vals = []
        ordering = pick_block_cg(objects, 'b0')
        for j in range(num_trials):
            stats = ancestral_sampling_acc(objects, ordering, num_ancestral_samples)
            vals.append(stats.get('FINAL', 0) / stats.get('START', 0))
        b1.append((np.mean(vals), np.std(vals)))

        vals = []
        ordering = pick_block_cg(objects, 'b1')
        for j in range(num_trials):
            stats = ancestral_sampling_acc(objects, ordering, num_ancestral_samples)
            vals.append(stats.get('FINAL', 0) / stats.get('START', 0))
        b2.append((np.mean(vals), np.std(vals)))
    return b1, b2


#%%
if __name__ == '__main__':
    N = 1000
    dataset = generate_dataset(N, 50)
    X, Y = dataset_to_XY(dataset)
    X_train, Y_train = X[:N // 2], Y[:N // 2]
    X_test, Y_test = X[N // 2:], Y[N // 2:]
#%%
    model = MLP([16, 16, 16, 2], 2)
    # criterion = torch.nn.L1Loss()
    # lossf = torch.nn.GaussianNLLLoss(reduction='mean')
    # def criterion(p,y):
    #     return lossf(p, y, p*(1.-p)/30.)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(200):
        train(model, optimizer, criterion, X_train, Y_train, batch_size=1000, epochs=2)
        train_loss, train_acc = eval(model, X_train, Y_train)
        loss, acc = eval(model, X_test, Y_test)
        print(f"loss: {train_loss} vs {loss}, acc: {train_acc} vs {acc}")

# %%

    model = MLP([16, 16, 16, 2], 2)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(50):
        train(model, optimizer, criterion, X_train, Y_train, batch_size=100, epochs=1)
        train_loss, train_acc = eval(model, X_train, Y_train)
        loss, acc = eval(model, X_test, Y_test)
        print(f"loss: {train_loss} vs {loss}, acc: {train_acc} vs {acc}")
# %%
