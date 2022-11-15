#%%
from learning.data_models import StreamInstanceClassifierV2Info
from torch_geometric.data import DataLoader, Batch
from learning.gnn.data import construct_problem_graph, construct_problem_graph_input
from learning.gnn.models import GraphNetwork, PLOIAblationModel
# %%
from glob import glob
import os
import pickle
import json
import torch
# %%
from torch_geometric.data import DataLoader, Batch
import time
import os
import numpy as np
import torch
import copy
import json
from tqdm import tqdm
from learning.pddlstream_utils import item_to_dict, ancestors, objects_from_facts
import pickle
import json

def train_model_graphnetwork(
    model,
    datasets,
    criterion,
    optimizer,
    step_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
):
    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_val_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    for e in range(epochs):

        running_loss = 0.
        running_num_samples = 0

        model.train()

        for i, d in enumerate(trainset):
            preds = model(d)
            loss = criterion(preds.flatten(), d.y)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()
            if (i % step_every == (step_every - 1)) or i == (len(trainset) - 1):
                optimizer.step()
                optimizer.zero_grad()

        print(f"== [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}")

        if e % save_every == (save_every - 1):

            model.eval()

            savefile = os.path.join(save_folder, f"model_{e:04d}.pt")
            torch.save(model.state_dict(), savefile)
            print(f"Saved model checkpoint {savefile}")

            _,_,losses = evaluate_dataset(model, criterion, validset)
            
            avg_loss = np.mean(losses)
            print(f"===== [EPOCH {e:03d} / {epochs}] Val Avg loss: {avg_loss:03.5f}")

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, "best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(f"Found new best model with avg loss {avg_loss} at epoch {e}. Saved!")

    time_elapsed = time.time() - since
    print(f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec")

    return best_seen_model_weights

def evaluate_dataset(model, criterion, dataset):
    losses = []
    model.eval()
    print("Starting Evaluation")
    datas = []
    for d in tqdm(dataset):
        preds = model(d)
        logit = torch.sigmoid(preds)
        loss = criterion(preds, d.y.unsqueeze(1)).detach().cpu().numpy()
        losses.append(loss)
    return None, None, losses

def load_pkl(fpath):
    if fpath.startswith('/jobs'):
        fpath = fpath.replace('/jobs', JOBS_HOME)
    with open(fpath, 'rb') as f:
        d = pickle.load(f)
    return d

def load_json(fpath):
    if fpath.startswith('/jobs'):
        fpath = fpath.replace('/jobs', JOBS_HOME)
    with open(fpath, 'r') as f:
        d = json.load(f)
    return d

def get_all_ancestors(objects, atom_map):
    req = set()

    for o in objects:

        for atom in atom_map:
            if o in atom and not any(o in f for f in atom_map[atom]):
                break
        else:
            raise ValueError
        req.update(objects_from_facts(ancestors(atom, atom_map)))
    return req

def get_labels(data):
    stats = load_json(data['stats_path'])
    atom_map = item_to_dict(stats['atom_map'])

    preimage_objects = {o for f in stats['solution'] for o in f[1]}
    extra_objects = get_all_ancestors(preimage_objects, atom_map)

    preimage_objects = preimage_objects | extra_objects
    goal_objects = {o for f in data['problem_info'].goal_facts for o in f[1:]}
    initial_objects = {o for f in data['problem_info'].initial_facts for o in f[1:]}

    necessary_objects = (preimage_objects & initial_objects) | goal_objects
    irrelevant_objects = initial_objects - preimage_objects
    
    return necessary_objects, irrelevant_objects

# %%
def load_data(data_files):
    X = []
    counts = dict(neg=0, pos=0)
    for f in data_files:
        try:
            f_data = load_pkl(f)
            pos, neg = get_labels(f_data)

            f_data['problem_info'].problem_graph = construct_problem_graph(f_data['problem_info'])
            x = construct_problem_graph_input(f_data['problem_info'], f_data['model_info']).to(device)
            y = []
            for d in x.nodes:
                if d in pos:
                    y.append(1.)
                    counts['pos'] += 1
                else:
                    assert d in neg
                    y.append(0.)
                    counts['neg'] += 1
            x.y = torch.tensor(y).to(device)
            X.append(x)
        except Exception as e:
            print(f"Warning: Skipping {f}. {e}")
            raise e
            pass
    print(counts)
    return X
# %%
if __name__ == '__main__':
    JOBS_HOME = '/home/mohammed/drake-tamp/jobs-feb/jobs'
    data_json, model_home = 'results-corl/blocksworld-dset.json', 'model_files/ploi'

    if not os.path.exists(model_home):
        os.makedirs(model_home, exist_ok=True)
    with open(data_json, 'r') as f:
        data_paths = json.load(f)

    train_files = ["jobs-feb" + f for f in data_paths['train']]
    np.random.shuffle(train_files)
    val_files = train_files[-30:]
    train_files = train_files[:-30]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # %%
    f_data = load_pkl(train_files[0])
    model_info = StreamInstanceClassifierV2Info(**f_data["model_info"].__dict__)
    model = PLOIAblationModel(model_info).to(device)
    datasets = dict(
        train=DataLoader(load_data(train_files), batch_size=32, shuffle=True),
        val=DataLoader(load_data(val_files), batch_size=32)
    )


    # %%
    lr = 0.0001
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model_graphnetwork(model, datasets, criterion=criterion, optimizer=optimizer, epochs=1000, save_folder=model_home)
