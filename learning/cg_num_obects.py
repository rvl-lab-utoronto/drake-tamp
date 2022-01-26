from learning.cg_toy_simple import *

def random(start, end):
    return (np.random.random() * (end - start)) + start

def set_placements(region, blocks):
    
    start = region['x']
    end = region['x'] + region['width']
    remaining = end - start
    min_required = sum(b['width'] for b in blocks.values()) + 1e-2

    if min_required > remaining:
        raise ValueError((min_required, remaining))

    block_names = list(blocks)
    np.random.shuffle(block_names)

    for block in block_names:
        blocks[block]['x'] = random(start, end - min_required)
        start = blocks[block]['x'] + blocks[block]['width']
        min_required = min_required - blocks[block]['width']


def generate_scene(heights=None):
    if heights is None:
        num_blocks = np.random.randint(2, 5)
        heights = [np.random.randint(1, 4) for block in range(num_blocks)]

    REGIONS = frozendict({
        "r1": {"width": 10, "x": 0, "y": -0.3, "height": .3},
#         "r2": {"width": 3, "x": 6, "y": -0.3, "height": .3}
    })

    GRIPPERS = {
        "g1": {"width": 3.8, "height": 1, "x": 2, "y": 8, "color": None}
    }


    colors = plt.get_cmap("tab10")
    BLOCKS = {
        f"b{i}": {
            "width": 2.45,
            "y": 0,
            "height": height,
            "color": colors(i)
        } for i, height in enumerate(heights)
    }
    set_placements(REGIONS['r1'], BLOCKS)

    return (WORLD, GRIPPERS, REGIONS, BLOCKS)

def pick_block_cg(scene, objects, block_name):
    return [
        StreamAction(grasp_stream, (objects['g1'].pddl, objects[block_name].pddl), ('?grasp',)),
        StreamAction(ik_stream, (objects['g1'].pddl, objects[block_name + '_pose'].pddl, '?grasp'), ('?conf',)),
    ] + [
        StreamAction(safety_stream, (objects['g1'].pddl, '?conf', objects[bname].pddl, objects[f'{bname}_pose'].pddl), (f'?safe_{bname}',)) 
        for bname in scene[-1] if bname != block_name
    ]


def generate_dataset(num_scenarios=1000, num_ancestral_samples=30):
    dataset = []
    for i in range(num_scenarios):
        scene = generate_scene()
        objects = scene_objects(*scene)
        
        for bname in scene[-1]:
            ordering = pick_block_cg(scene, objects, bname)
            stats = ancestral_sampling_acc(objects, ordering, num_ancestral_samples)
            dataset.append((scene, bname, ordering, stats))

    return dataset


## model stuff
def block_features(bname, scene):
    return [scene[-1][bname]['x'], scene[-1][bname]['height']]

def MLP(layers, input_dim):
    mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(torch.nn.LeakyReLU())
        mlp_layers.append(torch.nn.Linear(layers[layer_num], layers[layer_num + 1]))

    return torch.nn.Sequential(*mlp_layers)


class MultiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = MLP([16, 2], 2) # embedder
        self.g = MLP([16, 3], 2) # represents combination of grasp and ik streams
        self.f = MLP([16, 16, 3], 4) # represents safety stream, first inputs represent the "conf" grasping an object, second represent potential blocker and its position
        self.all = MLP([16, 16, 1], 2)
    def forward(self, to_grasp, others):
#         to_grasp, other = x[:,0,:], x[:,1,:]
        h_to_grasp = self.h(to_grasp)
        h_conf = self.g(h_to_grasp)
        h_conf, p_conf = h_conf[:-1], h_conf[-1:]
        p = p_conf
        out = 0
        num = 0
        for other in others:
            num += 1
            h_other = self.h(other)
            f_other = self.f(torch.cat([h_conf, h_other]))
            f_other, p_other = f_other[:-1], f_other[-1:]
            p = torch.cat([p, p_other])
            out += f_other
        out = self.all(out / num)
        p = torch.cat([p, out])
        p = torch.sigmoid(p)
        return p

    def pred(model, X):
        model.eval()
        p = []
        for (o1, others) in X:
            out = model(torch.Tensor(o1), map(torch.Tensor, others))
            p.append(out.detach().numpy()[-1])
        p = np.array(p)
        return p


    def evaluate(model, X, Y):
        model.eval()
        p = []
        for (o1, others), y in zip(X, Y):
            out = model(torch.Tensor(o1), map(torch.Tensor, others))
            p.append(out.detach().numpy()[-1])
        Y = np.array([y[-1] for y in Y])
        p = np.array(p)
        return np.abs(p - Y)

    def dataset_to_XY(self, dataset):
        X = []
        Y = []
        
        for i in range(0, len(dataset)):
            (scene, block, ordering, stats) = dataset[i]
            to_pick = block_features(block, scene)
            others = [block_features(b, scene) for b in scene[-1] if b != block]
            X.append((to_pick, others))

            Ys = []
            for stream_action in ordering[1:]:
                Ys.append(stats.get(stream_action, 0) / stats['START'])
            assert all(act.outputs[0] == f"?safe_{b}" for b, act in zip([b for b in scene[-1] if b != block], ordering[2:]))
            Ys.append(stats.get('FINAL', 0) / stats['START'])
            Y.append(Ys)


        return X, Y

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = MLP([16, 2], 2) # embedder
        self.g = MLP([16, 2], 2) # represents combination of grasp and ik streams
        self.f = MLP([16, 16, 2], 4) # represents safety stream, first inputs represent the "conf" grasping an object, second represent potential blocker and its position
        self.all = MLP([16, 16, 1], 2)
    def forward(self, to_grasp, others):
#         to_grasp, other = x[:,0,:], x[:,1,:]
        h_to_grasp = self.h(to_grasp)
        h_conf = self.g(h_to_grasp)
        out = 0
        num = 0
        for other in others:
            num += 1
            h_other = self.h(other)
            out += self.f(torch.cat([h_conf, h_other]))
        out = self.all(out / num)
        out = torch.sigmoid(out)
        return out

    def pred(model, X):
        model.eval()
        p = []
        for (o1, others) in X:
            out = model(torch.Tensor(o1), map(torch.Tensor, others))
            p.append(out.detach().numpy())
        p = np.array(p)
        return p

    def evaluate(model, X, Y):

        model.eval()
        p = []
        for (o1, others), y in zip(X, Y):
            out = model(torch.Tensor(o1), map(torch.Tensor, others))
            p.append(out.detach().numpy())
        Y = np.array(Y)
        p = np.array(p)
        assert Y.shape == p.shape
        return np.abs(p - Y)

    def dataset_to_XY(self, dataset):
        X = []
        Y = []
        
        for i in range(0, len(dataset)):
            (scene, block, ordering, stats) = dataset[i]
            to_pick = block_features(block, scene)
            others = [block_features(b, scene) for b in scene[-1] if b != block]
                
            X.append((to_pick, others))
            Y.append([stats.get('FINAL', 0) / stats['START']])


        return X, Y

def train(model, criterion, optimizer, X, Y, batch_size=100, epochs=10):
    batch_size = min(batch_size, len(X))
    model.train()
    for _ in range(epochs):
        total_loss = 0
        for i, ((o1, others), y) in enumerate(zip(X, Y)):
            out = model(torch.Tensor(o1), map(torch.Tensor, others))
            y = torch.Tensor(y)
            assert y.shape == out.shape
            loss = criterion(out, y)
            loss.backward()
            total_loss += loss.item()
            if i % batch_size == (batch_size - 1) or (i == len(X) - 1):
                optimizer.step()
                optimizer.zero_grad()

def train_eval(model, train_dataset, test_dataset, criterion, optimizer, num_epochs = 500, eval_every = 10):
    X_train, Y_train = model.dataset_to_XY(train_dataset)
    X_test, Y_test = model.dataset_to_XY(test_dataset)
    train_loss, test_loss = [], []
    for i in range(1 + (num_epochs // eval_every)):
        train_l1 = model.evaluate(X_train, Y_train).mean()
        test_l1 = model.evaluate(X_test, Y_test).mean()
        train_loss.append(train_l1)
        test_loss.append(test_l1)
        if i < (num_epochs // eval_every):
            train(model, criterion, optimizer, X_train, Y_train, epochs=eval_every)
        print(f"loss: {train_l1} vs {test_l1}")
    return train_loss, test_loss

#%%
if __name__ == '__main__':
    K = 30
    N_train = 100
    N_test = 100
    train_dataset = generate_dataset(N_train, K)
    test_dataset = generate_dataset(N_test, K)


    model_A = Model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_A.parameters(), lr=0.01)
    train_loss_A, test_loss_A = train_eval(model_A, train_dataset, test_dataset, criterion, optimizer)

    model_B = MultiModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_B.parameters(), lr=0.01)
    train_loss_B, test_loss_B = train_eval(model_B, train_dataset, test_dataset, criterion, optimizer)

#%%