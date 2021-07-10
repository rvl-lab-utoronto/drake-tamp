import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pad_sequence

class StreamInstanceClassifier(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, stream_num_domain_facts, num_predicates, object_node_feature_size, feature_size=8, lstm_size=10,  mlp_out=1, use_gcn=True, use_object_model=True):
        super(StreamInstanceClassifier, self).__init__()
        self.use_object_model = use_object_model
        self.use_gcn = use_gcn
        if self.use_object_model:
            self.fact_model = FactModel(num_predicates, input_size=lstm_size, hidden_size=lstm_size)
            self.object_network = SimpleGCN(object_node_feature_size, lstm_size)
            node_feature_size += lstm_size

        if self.use_gcn:
            self.graph_network = SimpleGCN(node_feature_size, feature_size)
        else:
            self.graph_network = GraphNetwork(
                node_feature_size=node_feature_size,
                edge_feature_size=edge_feature_size,
                hidden_size=feature_size,
            )
        self.mlps = tuple([MLP([16, mlp_out], feature_size*s)  for s in stream_num_domain_facts])
        for i, mlp in enumerate(self.mlps):
            setattr(self, f'mlp{i}', mlp)


    def forward(self, data, score=False):
        if self.use_object_model:
            object_representations = self.object_network(data.objects_data)
            fact_representations = self.fact_model(data, object_representations)
            data.z = torch.cat((data.x, fact_representations), dim=1)
            x = self.graph_network(data, attr='z')
        else:
            x = self.graph_network(data, attr='x')

        cand = data.candidate
        assert cand[0] > 0, "Considering an initial condition?"
        stream_mlp = self.mlps[cand[0] - 1]
        domain_facts = torch.cat([x[p] for p in  cand[1:]])
        x = stream_mlp(domain_facts)
        if score:
            return torch.sigmoid(x)
        return x

class EdgeModel(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_size, dropout=0.0):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_size + edge_feature_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, src, dst, edge_attr, u=None, batch=None):
        # src, dst: [E, F_x], where E is num edges, F_x is node-feature dimensionality
        # edge_attr: [E, F_e], where E is num edges, F_e is edge-feature dimensionality
        out = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, node_feature_size, hidden_size, n_targets, dropout=0.0):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(node_feature_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(node_feature_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, n_targets),
        )

    def forward(self, x, edge_idx, edge_attr, u=None, batch=None):
        row, col = edge_idx
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GraphNetwork(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_size, dropout=0.0):
        super(GraphNetwork, self).__init__()
        self.meta_layer_1 = self.build_meta_layer(node_feature_size, edge_feature_size, hidden_size, hidden_size, dropout=dropout)
        self.meta_layer_2 = self.build_meta_layer(hidden_size, hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.meta_layer_3 = self.build_meta_layer(hidden_size, hidden_size, hidden_size, hidden_size, dropout=dropout)

    def build_meta_layer(self, node_feature_size, edge_feature_size, hidden_size, n_targets, dropout=0.0):
        return MetaLayer(
            edge_model=EdgeModel(node_feature_size, edge_feature_size, hidden_size, dropout=dropout),
            node_model=NodeModel(node_feature_size, hidden_size, n_targets, dropout=dropout),
        )

    def forward(self, data, attr='x'):
        x, edge_idx, edge_attr = getattr(data, attr), data.edge_index, data.edge_attr
        x, edge_attr, _ = self.meta_layer_1(x, edge_idx, edge_attr)
        x, edge_attr, _ = self.meta_layer_2(x, edge_idx, edge_attr)
        x, edge_attr, _ = self.meta_layer_3(x, edge_idx, edge_attr)
        return x

class SimpleGCN(nn.Module):
    def __init__(self, node_feature_size, feature_size):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(node_feature_size, feature_size)
        self.conv2 = GCNConv(feature_size, feature_size)
        self.conv3 = GCNConv(feature_size, feature_size)
    def forward(self, data, attr = 'x'):
        x, edge_index = getattr(data, attr), data.edge_index

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x

def MLP(layers, input_dim, dropout=0.0):
    mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(torch.nn.LeakyReLU())
        mlp_layers.append(torch.nn.Linear(layers[layer_num], layers[layer_num + 1]))
    # if len(layers) > 1:
    #     mlp_layers.append(torch.nn.LayerNorm(mlp_layers[-1].weight.size()[:-1]))
    #     if dropout > 0:
    #         mlp_layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*mlp_layers)


class FactModel(nn.Module):
    def __init__(self, num_predicates, input_size=4, hidden_size=4):
        super(FactModel, self).__init__()
        self.predicate_embedding = nn.Embedding(num_predicates, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=1)
    def forward(self, data, object_embeddings):
        preds = self.predicate_embedding(torch.tensor([n[:1] for n in data.nodes_ind]))
        fact_tokens = []
        for i in range(len(data.nodes_ind)):
            n = data.nodes_ind[i]
            if len(n) > 1:
                objs = [object_embeddings[i] for i in n[1:]]
                fact_tokens.append(torch.stack([preds[i].squeeze(0)] + objs))
            else:
                fact_tokens.append(preds[i])

        x = pad_sequence(fact_tokens)

        x, _ = self.lstm(x)
        x = x[-1] # select last output
        return x
