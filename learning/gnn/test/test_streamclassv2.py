#%%
import pandas
import pickle
import numpy as np
import json
import os
from tqdm import tqdm
import torch
from torch_geometric.data.dataloader import DataLoader
from learning.gnn.data import (
    DeviceAwareLoaderWrapper,
    construct_stream_classifier_input_v2,
    Dataset,
    TrainingDataset,
    EvaluationDatasetSampler,
    get_base_datapath,
)
from learning.gnn.models import (
    StreamInstanceClassifierV2,
)
from learning.data_models import StreamInstanceClassifierV2Info
from learning.gnn.train import evaluate_model
import matplotlib.pyplot as plt

#%%

base_datapath = get_base_datapath()

with open(os.path.expanduser("~/drake-tamp/learning/data/experiments/kitchen_less_axioms.json"), "r") as f:
    exp_list = json.load(f)
train_files = [os.path.join(base_datapath, d) for d in exp_list["train"]]
val_files = [os.path.join(base_datapath, d) for d in exp_list["validation"]]
meta_data = [pickle.load(open(f, "rb")) for f in val_files]

device = torch.device("cuda")
model_info_class = StreamInstanceClassifierV2Info
valset = Dataset(
    construct_stream_classifier_input_v2,
    model_info_class,
    preprocess_all=False
)

valset.from_pkl_files(*val_files)
valset.prepare()
val_sampler = EvaluationDatasetSampler(valset)
val_loader = DataLoader(
    valset,
    sampler=val_sampler,
    batch_size=1,
    num_workers=8,
)
model = StreamInstanceClassifierV2(valset.model_info, feature_size = 32, hidden_size = 32)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=3 * torch.ones([1]))
criterion.to(device)
model.to(device)
model.load_state_dict(torch.load(os.path.join(os.path.expanduser("~/drake-tamp/model_files/streamclassv2_kitchen_less_axioms"), "best.pt")))

#evaluate_model(
    #model,
    #criterion,
    #DeviceAwareLoaderWrapper(val_loader, device),
    #save_path="./"
#)
loader = DeviceAwareLoaderWrapper(val_loader, device)

#%%

logits = {}
labels = {}
losses = {}
datas_list = {}
model.eval()
print("Starting Evaluation")

for d in tqdm(loader):
    preds = model(d)
    problem_keys = d.problem_index # this is a list of single element lists
    logit = torch.sigmoid(preds)
    for row_index, problem_key in enumerate(problem_keys):
        problem_key = problem_key[0]
        loss = criterion(preds[row_index], d.y[row_index].unsqueeze(dim=0))
        losses.setdefault(problem_key, []).append(loss.detach().cpu().numpy().item())
        logits.setdefault(problem_key, []).append(logit[row_index].detach().cpu().numpy().item())
        labels.setdefault(problem_key, []).append(d.y[row_index].detach().cpu().numpy().item())
        datas_list.setdefault(problem_key, []).append(d)

#%%
print("Starting analysis")

dfs = {}
streams = set()

unmerged = []

for (problem_key, data), meta in zip(datas_list.items(), meta_data):
    df = pandas.DataFrame()
    meta_df = pandas.DataFrame(meta)
    los = losses[problem_key]
    logs = logits[problem_key]
    labs = labels[problem_key]
    datas = datas_list[problem_key]
    df["losses"] = los
    df["logits"] = logs
    df["labels"] = labs
    seds = [d.stream_schedule[0] for d in datas]
    streams = set([r["name"] for sed in seds for r in sed]) | streams
    df["stream_schedules"] = seds 
    res = [sed[-1] for sed in seds]
    df["stream_results"] = res
    sts = [r["name"] for r in res]
    df["stream_name"] = sts
    #df = df.sort_values("losses", ascending = False)
    unmerged.append((df, meta))
    dfs[problem_key] = df

stream_dfs = {}


for l in [0.0, 1.0]:
    for stream in streams:
        fig, ax = plt.subplots()
        df = pandas.concat([d[d["stream_name"] == stream] for d in dfs.values()])
        df = df[np.isclose(df["labels"], l)]
        #print(df["losses"])
        ax.hist(df["logits"], bins = 40, alpha = 0.5, label = stream)
        stream_dfs[stream] = df

        ax.legend()
        ax.set_xlabel("logits")
        ax.set_ylabel("Occurances")
        ax.set_xlim([0,1])
        if np.isclose(l, 0.0):
            ax.set_title("Negative Examples")
            plt.savefig(f"{stream}_logits_histogram_neg.png", dpi = 400)
        else:
            ax.set_title("Positive Examples")
            plt.savefig(f"{stream}_logits_histogram_pos.png", dpi = 400)
        plt.close("all")

# %%

def sub_fact(fact, mapping):
    res = []
    for thing in fact:
        if thing in mapping:
            res.append(mapping[thing])
        else:
            res.append(thing)
    return tuple(res)

def sub_res(res, mapping):
    for ind, i in enumerate(res["input_objects"]):
        if i in mapping:
            res["input_objects"][ind] = mapping[i]


for df, meta in unmerged:
    problem_info = meta["problem_info"]
    object_mapping = problem_info.object_mapping
    initial_facts = [sub_fact(f, object_mapping) for f in problem_info.initial_facts]
    goal_facts = [sub_fact(f, object_mapping) for f in problem_info.goal_facts]
    for sched in df["stream_schedules"]:
        for res in sched:
            sub_res(res, object_mapping)

# %%
