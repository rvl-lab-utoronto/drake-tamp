import time
import os
import numpy as np
import torch
import copy


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


if __name__ == '__main__':
    from learning.gnn.data import parse_labels
    from learning.gnn.models import StreamInstanceClassifier
    dataset, model_info = parse_labels('/home/mohammed/drake-tamp/learning/data/labeled/2021-07-06-14:24:22.064.pkl')
    dataset2, _ = parse_labels('/home/mohammed/drake-tamp/learning/data/labeled/2021-07-06-14:54:42.372.pkl')
    dataset += dataset2
    model = StreamInstanceClassifier(
        node_feature_size=model_info.node_feature_size,
        edge_feature_size=model_info.edge_feature_size,
        stream_input_sizes=model_info.stream_input_sizes[1:],
        num_predicates=len(model_info.predicates),
        object_node_feature_size=model_info.object_node_feature_size,
        lstm_size=5, feature_size=4, use_gcn=False, use_object_model=True)
    neg = [d for d in dataset if d.y[0] == 0]
    pos = [d for d in dataset if d.y[0] == 1]
    data = dict(train=StratifiedRandomSampler(pos, neg, prop=0.5), val=StratifiedRandomSampler(pos, neg, prop=0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=.8*torch.ones([1]))
    train_model_graphnetwork(model, data, criterion=criterion, optimizer=optimizer, save_every=10, epochs=100)