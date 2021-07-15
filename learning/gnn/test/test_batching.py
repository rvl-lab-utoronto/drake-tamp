#%%

from learning.data_models import StreamInstanceClassifierInfo
from learning.gnn.data import EvaluationDataset, construct_input, HyperModelInfo, TrainingDataset, Dataset, construct_hypermodel_input, construct_with_problem_graph, get_base_datapath, get_pddl_key, query_data
from learning.gnn.models import HyperClassifier, StreamInstanceClassifier
from learning.gnn.train import evaluate_model, train_model_graphnetwork
from functools import partial
import torch
from torch_geometric.data import Batch, DataLoader
# %%

use_problem_graph = True
train_files = ["/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-14-02:11:35.009.pkl"]
input_fn = partial(construct_hypermodel_input, reduced = False)
if use_problem_graph:
    input_fn = construct_with_problem_graph(input_fn)
model_info_class = HyperModelInfo
model_fn = lambda model_info: HyperClassifier(
    model_info,
    with_problem_graph=use_problem_graph
)
trainset = TrainingDataset(
    input_fn,
    model_info_class,
    augment=True,
    stratify_prop=None,
    preprocess_all=False
)
trainset.from_pkl_files(*train_files)
trainset.prepare()
#%%

data = []
for i,d in enumerate(trainset):
    data.append(d)

batch = Batch().from_data_list(data)

#%%

model = model_fn(trainset.model_info)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=3*torch.ones([1]))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%
batch.batch

loader = DataLoader([d for d in trainset], batch_size = 32)

for e in range(100):
    model.train()

    for i,d in enumerate(loader):
        print(d)
        preds = model(d)
