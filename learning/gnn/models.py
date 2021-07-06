import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GCNConv

class StreamInstanceClassifier(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, stream_input_sizes, feature_size=8, mlp_out=1, use_gcn=False):
        super(StreamInstanceClassifier, self).__init__()
        if use_gcn:
            self.graph_network = SimpleGCN(node_feature_size, feature_size)
        else:
            self.graph_network = GraphNetwork(
                node_feature_size=node_feature_size,
                edge_feature_size=edge_feature_size,
                hidden_size=feature_size,
            )
        self.mlps = tuple([MLP([feature_size, mlp_out], feature_size * s) for s in stream_input_sizes])
        for i, mlp in enumerate(self.mlps):
            setattr(self, f'mlp{i}', mlp)


    def forward(self, data):
        x = self.graph_network(data)
        # assert len(data.candidate) == 1, "Cant support batching yet"
        cand = data.candidate
        assert cand[0] > 0, "Considering an initial condition?"
        mlp = self.mlps[cand[0] - 1]
        parents = torch.cat([x[p] for p in  cand[1:]])
        x = mlp(parents)
        if self.training:
            return x
        return torch.sigmoid(x)

class EdgeModel(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_size, dropout=0.0):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_size + edge_feature_size, hidden_size),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(node_feature_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, n_targets),
        )
    
    def forward(self, x, edge_idx, edge_attr, u=None, batch=None):
        row, col = edge_idx
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
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
    
    def forward(self, data):
        x, edge_idx, edge_attr = data.x, data.edge_index, data.edge_attr
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
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x

def MLP(layers, input_dim, dropout=0.0):
    mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers) - 1):
        mlp_layers.append(torch.nn.ReLU())
        mlp_layers.append(torch.nn.Linear(layers[layer_num], layers[layer_num + 1]))
    if len(layers) > 1:
        mlp_layers.append(torch.nn.LayerNorm(mlp_layers[-1].weight.size()[:-1]))
        if dropout > 0:
            mlp_layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*mlp_layers)
