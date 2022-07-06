from learning.gnn.metrics import accuracy, generate_figures, precision_recall
from learning.data_models import StreamInstanceClassifierInfo
from learning.gnn.data import construct_input
from learning.gnn.models import StreamInstanceClassifier
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
    evaluate_model,
    step_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
):
    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_validation = -np.inf

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

            val_eval = evaluate_model(model, criterion, validset, save_path = save_folder)
            print(f"===== [EPOCH {e:03d} / {epochs}] Val: {val_eval:03.5f}")

            if val_eval > best_seen_validation:
                best_seen_validation = val_eval
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, "best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(f"Found new best model with val {val_eval} at epoch {e}. Saved!")

    time_elapsed = time.time() - since
    print(f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec")

    return best_seen_model_weights

class StratifiedRandomSampler:
    def __init__(self, pos, neg, prop=0.5, epoch_size=200):
        self.pos = pos
        self.neg = neg
        self.prop = prop
        self.epoch_size = epoch_size
        self.i = None
    def __iter__(self):
        self.i = 0
        return self
    def __len__(self):
        return self.epoch_size
    def __next__(self):
        self.i += 1
        if self.i > self.epoch_size:
            raise StopIteration
        if np.random.random() < self.prop:
            return self.pos[np.random.choice(len(self.pos))]
        else:
            return self.neg[np.random.choice(len(self.neg))]

def evaluate_model_loss(model, criterion, dataset, save_path=None):
    running_loss = 0
    running_num_samples = 0

    for d in dataset:
        preds = model(d)
        preds = model(d)
        loss = criterion(preds.flatten(), d.y)
        running_loss += loss.item()
        running_num_samples += 1

    return -running_loss / running_num_samples

    
def evaluate_dataset(model, criterion, dataset):
    logits = {}
    labels = {}
    losses = {}
    model.eval()
    print("Starting Evaluation")
    datas = []
    for d in tqdm(dataset):
        problem_keys = d.problem_index # this is a list of single element lists
        preds = model(d)
        logit = torch.sigmoid(preds)
        datas.append(d)
        for row_index, problem_key in enumerate(problem_keys):
            problem_key = problem_key[0]
            loss = criterion(preds[row_index], d.y[row_index].unsqueeze(dim=0))
            losses.setdefault(problem_key, []).append(loss.detach().cpu().numpy().item())
            logits.setdefault(problem_key, []).append(logit[row_index].detach().cpu().numpy().item())
            labels.setdefault(problem_key, []).append(d.y[row_index].detach().cpu().numpy().item())
    return logits, labels, losses

def evaluate_model_stream(model, criterion, dataset, save_path=None):
    problem_logits, problem_labels, problem_losses = evaluate_dataset(model, criterion, dataset)
    problem_stats = {}
    pct_excluded = []
    per_problem_loss = []
    for problem_key in problem_logits:
        logits, labels, losses = problem_logits[problem_key], problem_labels[problem_key], problem_losses[problem_key]
        thresholds, precision, positive_recall, negative_recall = precision_recall(logits, labels)

        index_of_total_recall = int(np.where(positive_recall == 1.)[0][0])
        accuracy_at_total_recall = float(accuracy(logits, labels, thresholds[index_of_total_recall]))
        stats = dict(
            losses=losses,
            thresholds=thresholds.tolist(),
            precision=precision.tolist(),
            positive_recall=positive_recall.tolist(),
            negative_recall=negative_recall.tolist(),
            index_of_total_recall=index_of_total_recall,
            accuracy_at_total_recall=accuracy_at_total_recall,
            logits=list(logits),
            labels = list(labels),
        )
        per_problem_loss.append(np.mean(losses))
        problem_stats[problem_key] = stats
        pct_excluded.append(negative_recall[index_of_total_recall])

    generate_figures(problem_stats, save_path)
    with open(os.path.join(save_path, 'stats.json'), 'w') as f:
        json.dump(problem_stats, f)
    
    overall_pct_excluded = np.mean(pct_excluded)
    overall_loss = np.mean(per_problem_loss)
    print(f"\Average Prop of irrelevant facts excluded: {overall_pct_excluded:.2f}")
    print(f"\Average loss: {overall_loss:.2f}")
    return overall_pct_excluded

    