from learning.gnn.metrics import accuracy, generate_figures, precision_recall
from learning.data_models import StreamInstanceClassifierInfo
from learning.gnn.data import construct_input
from learning.gnn.models import StreamInstanceClassifier
import time
import os
import numpy as np
import torch
import copy
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
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    for e in range(epochs):

        running_loss = 0.
        running_num_samples = 0

        model.train()

        for i, d in enumerate(trainset):
            preds = model(d)
            loss = criterion(preds, d.y)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()
            if (i % step_every == (step_every - 1)) or i == (len(trainset) - 1):
                optimizer.step()
                optimizer.zero_grad()

        print(f"== [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}")

        if e % save_every == 0:

            model.eval()

            savefile = os.path.join(save_folder, f"model_{e:04d}.pt")
            torch.save(model.state_dict(), savefile)
            print(f"Saved model checkpoint {savefile}")

            running_loss = 0.
            running_num_samples = 0

            for d in validset:
                preds = model(d)
                loss = criterion(preds, d.y)

                running_loss += loss.item()
                running_num_samples += 1

            print(f"===== [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}")

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, "best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!")

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

def evaluate_dataset(model, dataset):
    logits = []
    labels = []
    model.eval()
    for d in dataset:
        logit = torch.sigmoid(model(d)).detach().numpy().item()
        logits.append(logit)
        labels.append(d.y.detach().numpy().item())
        if len(logits) == 100:
            break
    logits = np.array(logits)
    labels = np.array(labels)
    return logits, labels

def evaluate_model(model, dataset, save_path=None):
    logits, labels = evaluate_dataset(model, dataset)
    thresholds, precision, positive_recall, negative_recall = precision_recall(logits, labels)

    index_of_total_recall = int(np.where(positive_recall == 1.)[0][0])
    accuracy_at_total_recall = float(accuracy(logits, labels, thresholds[index_of_total_recall]))
    stats = dict(
        thresholds=thresholds.tolist(),
        precision=precision.tolist(),
        positive_recall=positive_recall.tolist(),
        negative_recall=negative_recall.tolist(),
        index_of_total_recall=index_of_total_recall,
        accuracy_at_total_recall=accuracy_at_total_recall,
    )
    generate_figures(stats, save_path)
    with open(os.path.join(save_path, 'stats.json'), 'w') as f:
        json.dump(stats, f)
    print(f"Total Recall Threshold: {thresholds[index_of_total_recall]:.2f}\nProportion of irrelevant facts excluded: {negative_recall[index_of_total_recall]:.2f}")

    