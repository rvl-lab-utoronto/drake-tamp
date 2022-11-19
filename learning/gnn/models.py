from math import factorial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import stream
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, MetaLayer
from torch_scatter import scatter_mean


def nPr(n, r):
    assert isinstance(n, int), "n must be an int"
    assert isinstance(r, int), "r must be an int"
    return factorial(n)//factorial(n-r)

class PLOIAblationModel(nn.Module):

    def __init__(
        self,
        model_info,
        feature_size=16,
        hidden_size=16,
    ):
        super().__init__()
        self.problem_graph_network = ProblemGraphNetwork(
            node_feature_size=model_info.problem_graph_node_feature_size,
            edge_feature_size=model_info.problem_graph_edge_feature_size,
            hidden_size=feature_size,
            graph_hidden_size=hidden_size,
            dropout=.5
        )
        self.pg_mlp = MLP([hidden_size, 1], hidden_size, dropout=0.5)


    def forward(self, data, object_reps=None, score=False, update_reps=False):
        rep_x = self.problem_graph_network(data, return_x=True)
        prob_x = self.pg_mlp(rep_x)
        return prob_x
            


class HyperClassifier(nn.Module):
    """
    Stream result classifier based on object hypergraph.

    params:
        node_feature_size: the length of the node feature vectors
        edge_feature_size: the length of the edge feature vectors
        stream_num_domain_facts: the number of domain facts of the stream
        stream_num_inputs: the number of inputs to the stream
        feature_size: the output feature size
    """
    def __init__(
        self,
        model_info,
        with_problem_graph=False,
        feature_size=16,
        problem_graph_output_size=16,
        problem_graph_hidden_size=4,
        mlp_out=1,
        use_gnns = True,
    ):
        self.use_gnns = use_gnns
        node_feature_size = model_info.node_feature_size
        edge_feature_size = model_info.edge_feature_size
        stream_domains = model_info.stream_domains[1:]
        stream_num_inputs = model_info.stream_num_inputs[1:]
        super(HyperClassifier, self).__init__()
        self.graph_network = GraphNetwork(
            node_feature_size = node_feature_size,
            edge_feature_size = edge_feature_size,
            hidden_size = feature_size
        )
        self.with_problem_graph = with_problem_graph
        if self.with_problem_graph:
            self.problem_graph_network = ProblemGraphNetwork(
                node_feature_size=model_info.problem_graph_node_feature_size,
                edge_feature_size=model_info.problem_graph_edge_feature_size,
                hidden_size=problem_graph_hidden_size,
                graph_hidden_size=problem_graph_output_size
            )
        else:
            problem_graph_output_size = 0
        assert len(stream_domains) == len(stream_num_inputs), "Inequal number of streams"
        self.stream_num_inputs = stream_num_inputs
        # each domain fact and stream input has its own input feature vector

        self.use_gnns = use_gnns
        node_inp_size = feature_size
        edge_inp_size = feature_size
        if not self.use_gnns:
            node_inp_size = node_feature_size
            edge_inp_size = edge_feature_size

        self.mlps = []
        for domain, num_inputs in zip(stream_domains, stream_num_inputs):
            inp_size = num_inputs*node_inp_size
            for fact in domain:
                if len(fact) == 2: # unary facts have one edge
                    inp_size += 1*edge_inp_size
                else:
                    # every non-unary fact has two edges (bidirectional)
                    inp_size += nPr(len(fact)- 1, 2)*edge_inp_size
            #inp_size *= feature_size
            inp_size += problem_graph_output_size
            self.mlps.append(
                MLP([16, mlp_out], inp_size)
            )

        for i, mlp in enumerate(self.mlps):
            setattr(self, f"mlp{i}", mlp)

        self.use_gnns = use_gnns
        if (not use_gnns) and (not with_problem_graph):
            raise NotImplementedError(
                "Currently using problem graph is not supported without GNN's"
            )

    def forward(self, data, score = False):
        # first get node and edge embeddings from GNN
        x, edge_attr = data.x, data.edge_attr
        if self.use_gnns:
            x, edge_attr = self.graph_network(data, return_edge_attr = True)
        assert hasattr(data, "batch"), "Must batch the data"
        if self.with_problem_graph:
            prob_rep = Batch().from_data_list(data.problem_graph)
            if self.use_gnns:
                prob_rep = self.problem_graph_network(prob_rep)
        # candidate object embeddings, and candidate fact embeddings to mlp

        # group batch by stream type 
        mlp_to_input = {}
        mlp_to_batch_inds = {}

        for i, cand in enumerate(data.candidate):
            assert cand[0] > 0, "Considering an initial condition"
            stream_ind = cand[0] - 1
            stream_num_inp = self.stream_num_inputs[stream_ind]
            input_node_inds = cand[1:1 + stream_num_inp]
            dom_edge_inds = cand[1+ stream_num_inp:]

            mlp = self.mlps[stream_ind]
            subgraph_node_inds = torch.where(data.batch == i)
            node_inp = x[subgraph_node_inds][input_node_inds]
            edge_inp = edge_attr[
                torch.where(
                    (data.edge_index[0] >= min(subgraph_node_inds[0]).item()) &
                    (data.edge_index[0] <= max(subgraph_node_inds[0]).item())
                )
            ][dom_edge_inds]
            inp = torch.cat(
                (prob_rep[i].unsqueeze(0), node_inp.reshape(1, -1), edge_inp.reshape(1, -1)), dim = 1
            )
            if mlp in mlp_to_input:
                mlp_to_input[mlp] = torch.cat((mlp_to_input[mlp], inp), dim = 0)
                mlp_to_batch_inds[mlp] = torch.cat(
                    (mlp_to_batch_inds[mlp], torch.tensor([i]))
                )
            else:
                mlp_to_input[mlp] = inp
                mlp_to_batch_inds[mlp] = torch.tensor([i])

        out = torch.zeros((len(data.candidate), 1), device = x.device)
        for mlp, inp in mlp_to_input.items():
            mlpout = mlp(inp)
            batch_inds = mlp_to_batch_inds[mlp]
            out[batch_inds] = mlpout

        if score:
            return torch.sigmoid(out)

        return out

class MultiHeadStreamMLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, feature_size, hidden_size=16):
        super().__init__()
        self.encoder = MLP([hidden_size], num_inputs*feature_size)
        self.scorer = MLP([hidden_size, 1], hidden_size)
        self.decoders = []
        for i in range(num_outputs):
            mlp = MLP([hidden_size, feature_size], hidden_size)
            setattr(self, f"decoder{i}", mlp)
            self.decoders.append(mlp)

    def forward(self, x):
        hidden = self.encoder(x)
        outputs = [decoder(hidden) for decoder in self.decoders]
        score = self.scorer(hidden)
        return score, outputs

class StreamInstanceClassifierV2(nn.Module):

    def __init__(
        self,
        model_info,
        feature_size=16,
        hidden_size=16,
        score_initial_objects=True,
        decrease_score_with_depth=True,
    ):
        super().__init__()
        self.score_initial_objects = score_initial_objects
        self.decrease_score_with_depth = decrease_score_with_depth
        self.problem_graph_network = ProblemGraphNetwork(
            node_feature_size=model_info.problem_graph_node_feature_size,
            edge_feature_size=model_info.problem_graph_edge_feature_size,
            hidden_size=feature_size,
            graph_hidden_size=hidden_size
        )
        stream_domains = model_info.stream_domains[1:]
        stream_num_inputs = model_info.stream_num_inputs[1:]
        stream_num_outputs = model_info.stream_num_outputs[1:]
        self.stream_to_index = model_info.stream_to_index
        assert len(stream_domains) == len(stream_num_inputs), "Inequal number of streams"
        self.stream_num_inputs = stream_num_inputs
        self.stream_num_outputs = stream_num_outputs

        self.mlps = []
        for num_inputs, num_outputs in zip(stream_num_inputs, stream_num_outputs):
            self.mlps.append(
                MultiHeadStreamMLP(num_inputs, num_outputs, feature_size=feature_size, hidden_size=hidden_size)
            )

        for i, mlp in enumerate(self.mlps):
            setattr(self, f"mlp{i}", mlp)

        self.pg_mlp = MLP([hidden_size, 1], hidden_size)

    def get_init_reps(self, problem_graph):
        problem_graph = Batch().from_data_list([problem_graph])
        rep_x = self.problem_graph_network(problem_graph, return_x=True)
        if self.score_initial_objects:
            prob_x = self.pg_mlp(rep_x)
        else:
            prob_x = 100. * torch.ones((rep_x.shape[0], 1), device=rep_x.device)
        object_reps = {name: {"rep": rep_x[i], "logit": prob_x[i]} for i,name in enumerate(problem_graph.nodes[0])}
        return object_reps

    def forward(self, data, object_reps=None, score=False, update_reps=False):
        stream_schedule = data.stream_schedule
        if object_reps is None:
            assert hasattr(data, 'problem_graph')
            problem_graph = data.problem_graph
            assert isinstance(problem_graph, list) and len(problem_graph) == 1, "Batching not supported"
            object_reps = self.get_init_reps(problem_graph[0])
        else:
            assert not hasattr(data, 'problem_graph')
        
        assert isinstance(stream_schedule, list) and len(stream_schedule) == 1, "Batching not supported"
        stream_schedule = stream_schedule[0]

        for stream in stream_schedule:
            stream_index = self.stream_to_index[stream["name"]] - 1
            stream_mlp = self.mlps[stream_index]
            
            stream_inputs = torch.cat([object_reps[n]["rep"] for n in stream["input_objects"]], dim=0)

            out, outputs = stream_mlp(stream_inputs)
            if self.decrease_score_with_depth:
                stream_logits = torch.cat([object_reps[n]["logit"] for n in stream["input_objects"]], dim=0)
                prev_log = (stream_logits*torch.softmax(-stream_logits, dim=0)).sum(0, keepdim=True)
                out = out + prev_log - torch.logsumexp(torch.cat([out, prev_log, torch.tensor([0.], device=out.device)], dim=0), dim=0)
            for out_name, out_rep in zip(stream["output_objects"], outputs):
                object_reps[out_name] = {"rep": out_rep, "logit": out}

        out = out.unsqueeze(0)
        # candidate object embeddings, and candidate fact embeddings to mlp
        if score:
            return torch.sigmoid(out)

        return out
            
class StreamInstanceClassifier(nn.Module):
    def __init__(self, model_info, feature_size=8, lstm_size=10,  mlp_out=1, use_gcn=True, use_object_model=True):
        node_feature_size = model_info.node_feature_size
        edge_feature_size = model_info.edge_feature_size
        stream_num_domain_facts = model_info.stream_num_domain_facts[1:]
        num_predicates = len(model_info.predicates)
        object_node_feature_size = model_info.object_node_feature_size

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
        self.mlps = tuple([MLP([32, mlp_out], feature_size*s)  for s in stream_num_domain_facts])
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
        source, dest = edge_idx
        out = torch.cat([x[source], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, dest, dim=0, dim_size=x.size(0))
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

    def forward(self, data, return_edge_attr = False, attr='x'):
        x, edge_idx, edge_attr = getattr(data, attr), data.edge_index, data.edge_attr
        x, edge_attr, _ = self.meta_layer_1(x, edge_idx, edge_attr)
        x, edge_attr, _ = self.meta_layer_2(x, edge_idx, edge_attr)
        x, edge_attr, _ = self.meta_layer_3(x, edge_idx, edge_attr)
        if return_edge_attr:
            return x, edge_attr
        return x

class GlobalModel(torch.nn.Module):
    def __init__(self, node_representation_size, edge_representation_size, graph_feature_size, graph_representation_size, dropout=0.0):
        super(GlobalModel, self).__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(node_representation_size + graph_feature_size, graph_representation_size),
            nn.LeakyReLU(),
            nn.LayerNorm(graph_representation_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(graph_representation_size, graph_representation_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if u is not None:
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        else:
            out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)

class GraphAwareNodeModel(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, graph_feature_size, output_size, dropout = 0.0):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(node_feature_size + edge_feature_size, output_size),
            nn.LeakyReLU(),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(output_size, output_size),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(node_feature_size + output_size + graph_feature_size, output_size),
            nn.LeakyReLU(),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch = None):
        cand, col = edge_index
        out = torch.cat([x[cand], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GraphAwareEdgeModel(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, graph_feature_size, hidden_size, dropout=0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_size + edge_feature_size + graph_feature_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, src, dst, edge_attr, u, batch=None):
        # src, dst: [E, F_x], where E is num edges, F_x is node-feature dimensionality
        # edge_attr: [E, F_e], where E is num edges, F_e is edge-feature dimensionality
        out = torch.cat(
            [src, dst, edge_attr, u[batch]],
            dim=1
        )
        return self.edge_mlp(out)


class ProblemGraphNetwork(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_size, graph_hidden_size, dropout=0.0):
        super().__init__()
        self.meta_layer_1 = MetaLayer(
            node_model=NodeModel(node_feature_size, hidden_size, hidden_size, dropout=dropout),
            edge_model=EdgeModel(node_feature_size, edge_feature_size, hidden_size, dropout=dropout),
            global_model=GlobalModel(node_representation_size=hidden_size, edge_representation_size=hidden_size, graph_feature_size=0, graph_representation_size=graph_hidden_size)
        )
        self.meta_layer_2 = MetaLayer(
            edge_model=GraphAwareEdgeModel(hidden_size, hidden_size, graph_hidden_size, hidden_size, dropout=dropout),
            node_model=GraphAwareNodeModel(hidden_size, hidden_size, graph_hidden_size, hidden_size, dropout=dropout),
            global_model=GlobalModel(node_representation_size=hidden_size, edge_representation_size=hidden_size, graph_feature_size=graph_hidden_size, graph_representation_size=graph_hidden_size)
        )
        self.meta_layer_3 = MetaLayer(
            edge_model=GraphAwareEdgeModel(hidden_size, hidden_size, graph_hidden_size, hidden_size, dropout=dropout),
            node_model=GraphAwareNodeModel(hidden_size, hidden_size, graph_hidden_size, hidden_size, dropout=dropout),
            global_model=GlobalModel(node_representation_size=hidden_size, edge_representation_size=hidden_size, graph_feature_size=graph_hidden_size, graph_representation_size=graph_hidden_size)
        )
    def forward(self, data, attr='x', return_x=False):
        x, edge_idx, edge_attr = getattr(data, attr), data.edge_index, data.edge_attr
        assert hasattr(data, "batch"), "Need to batch the data"
        x, edge_attr, u = self.meta_layer_1(x, edge_idx, edge_attr, u=None, batch = data.batch)
        x, edge_attr, u = self.meta_layer_2(x, edge_idx, edge_attr, u=u, batch = data.batch)
        x, edge_attr, u = self.meta_layer_3(x, edge_idx, edge_attr, u=u, batch = data.batch)
        if return_x:
            return x
        return u

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
