import torch
import torch.nn.functional as F
import numpy as np
from learning.gnn.data import parse_labels
from learning.gnn.models import StreamInstanceClassifier
import numpy as np
from random import sample

def acc(model, dataset, sample_count=100):
    acc_on_pos = 0
    for data in sample(dataset, min(sample_count, len(dataset))):
        out = model(data)
        acc_on_pos += int(data.y.detach().numpy()[0] == np.argmax(np.exp(out.detach().numpy())))
    return acc_on_pos / len(dataset)

if __name__ == '__main__':
    dataset, model_info = parse_labels('/home/mohammed/drake-tamp/learning/data/labeled/2021-07-05-14:35:28.341.pkl')
    model = StreamInstanceClassifier(model_info.node_feature_size, model_info.edge_feature_size, model_info.stream_input_sizes[1:], feature_size=4, use_gcn=False)


    neg = [d for d in dataset if d.y[0] == 0]
    pos = [d for d in dataset if d.y[0] == 1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    batch_size = 32
    losses = []
    model.eval()
    print('Accuracy on pos', acc(model, pos))
    print('Accuracy on neg', acc(model, neg))

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        for i in range(len(dataset)):
            if i % 2 == 0:
                idx = np.random.choice(len(neg))
                data = neg[idx]
            else:
                idx = np.random.choice(len(pos))
                data = pos[idx]

            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            if i % batch_size == (batch_size - 1):
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.detach().numpy())
        model.eval()
        print('Accuracy on pos', acc(model, pos))
        print('Accuracy on neg', acc(model, neg))
