#%%
from torch_geometric.data import DataLoader, Batch
from learning.gnn.data import construct_problem_graph, construct_problem_graph_input
from learning.gnn.models import GraphNetwork
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
# %%
def load_data(data_files):
    X = []
    counts = dict(neg=0, pos=0)
    for f in data_files:
        with open(f, 'rb') as fb:
            f_data = pickle.load(fb)
        with open(os.path.join(os.path.dirname(f), 'stats.json'), 'r') as fb:
            stats_data = json.load(fb)
        del f_data['labels']
        f_data['problem_info'].problem_graph = construct_problem_graph(f_data['problem_info'])
        x = construct_problem_graph_input(f_data['problem_info'], f_data['model_info'])
        y = []
        for d in x.nodes:
            if any(d in fact for fact in stats_data['last_preimage']):
                y.append(1.)
                counts['pos'] += 1
            else:
                y.append(0.)
                counts['neg'] += 1
        x.y = torch.tensor(y)
        X.append(x)
    print(counts)
    return X
# %%
if __name__ == '__main__':
    import sys
    data_json, model_home = sys.argv[1:]
    if not os.path.exists(model_home):
        os.makedirs(model_home, exist_ok=True)
    with open(data_json, 'r') as f:
        data_paths = json.load(f)

    train_files = data_paths['train']
    val_files = data_paths['validation']
    # %%
    with open(train_files[0], 'rb') as fb:
        f_data = pickle.load(fb)
        num_predicates = f_data['model_info'].num_predicates
    problem_graph_node_feature_size = 3 + 1
    problem_graph_edge_feature_size = num_predicates + 1

    model = GraphNetwork(problem_graph_node_feature_size, problem_graph_edge_feature_size, 8, 1)

    datasets = dict(train=DataLoader(load_data(train_files), batch_size=1), val=DataLoader(load_data(val_files), batch_size=1))


    # %%
    pos_weight = 1/20
    lr = 0.001
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight * torch.ones([1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model_graphnetwork(model, datasets, criterion=criterion, optimizer=optimizer, epochs=300, save_folder=model_home)
