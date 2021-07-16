
import argparse
import json
import os

from torch_geometric.data.dataloader import DataLoader
from learning.data_models import StreamInstanceClassifierInfo
from learning.gnn.data import DeviceAwareLoaderWrapper, EvaluationDatasetSampler, TrainingDatasetSampler, construct_input, HyperModelInfo, TrainingDataset, Dataset, construct_hypermodel_input, construct_with_problem_graph, get_base_datapath, get_pddl_key, query_data
from learning.gnn.models import HyperClassifier, StreamInstanceClassifier
from learning.gnn.train import evaluate_model, train_model_graphnetwork
from functools import partial
import torch

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-home",
        type=str,
        default='/tmp'
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hyper", "streamclass"],
        default="hyper"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=3.
    )
    # parser.add_argument(
    #     "--augment-data",
    #     action="store_true"
    # )
    parser.add_argument(
        "--stratify-train-prop",
        type=float,
        default=None
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float
    )
    parser.add_argument(
        "--gradient-batch-size",
        default=1,
        type=int
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int
    )
    parser.add_argument(
        "--num-preprocessors",
        default=8,
        type=int
    )
    parser.add_argument(
        '--from-best',
        action="store_true"
    )
    parser.add_argument(
        '--use-reduced-graph',
        action="store_true"
    )
    parser.add_argument(
        '--use-problem-graph',
        action="store_true"
    )
    parser.add_argument(
        '--datafile',
        type=str,
        help="A path to a json file containing train and validation keys wich have a list of pkl paths as values."
    )
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    return parser


if __name__ == '__main__':
    args = make_argument_parser().parse_args()
    base_datapath = get_base_datapath()
    with open(args.datafile, 'r') as f:
        data = json.load(f)
    train_files = [os.path.join(base_datapath, d) for d in data['train']]
    val_files = [os.path.join(base_datapath, d) for d in data['validation']]

    if args.debug:
        train_files = train_files[:1]
        val_files = train_files[:1]

    if not os.path.exists(args.model_home):
        os.makedirs(args.model_home, exist_ok=True)

    if args.model == 'hyper':
        input_fn = partial(construct_hypermodel_input, reduced = args.use_reduced_graph)
        if args.use_problem_graph:
            input_fn = construct_with_problem_graph(input_fn)
        model_info_class = HyperModelInfo
        model_fn = lambda model_info: HyperClassifier(
            model_info,
            with_problem_graph=args.use_problem_graph
        )
    elif args.model == 'streamclass':
        input_fn = construct_input
        model_info_class = StreamInstanceClassifierInfo
        model_fn = lambda model_info: StreamInstanceClassifier(
            model_info,
            use_gcn=True,
            use_object_model=False
        )
    else:
        raise ValueError


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    valset = Dataset(
        input_fn,
        model_info_class,
        preprocess_all=False,
        max_per_run=200,
    )
    valset.from_pkl_files(*val_files)
    valset.prepare()
    val_sampler = EvaluationDatasetSampler(valset)
    val_loader = DataLoader(valset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_preprocessors)

    model = model_fn(valset.model_info)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=args.pos_weight*torch.ones([1]))


    criterion.to(device)
    model.to(device)

    if args.from_best:
        model.load_state_dict(torch.load(os.path.join(args.model_home, 'best.pt')))

    if not args.test_only:
        trainset = TrainingDataset(
            input_fn,
            model_info_class,
            preprocess_all=False,
        )
        trainset.from_pkl_files(*train_files)
        trainset.prepare()
        train_sampler = TrainingDatasetSampler(trainset, epoch_size=200, stratify_prop=args.stratify_train_prop)
        train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_preprocessors)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model_graphnetwork(
            model,
            dict(
                train=DeviceAwareLoaderWrapper(train_loader, device),
                val=DeviceAwareLoaderWrapper(val_loader, device)
            ),
            criterion=criterion,
            optimizer=optimizer,
            step_every=args.gradient_batch_size,
            save_every=args.save_every,
            epochs=args.epochs,
            save_folder=args.model_home
        )
        # Load the best checkoibt for evaluation
        model.load_state_dict(torch.load(os.path.join(args.model_home, 'best.pt')))
 
    evaluate_model(model, criterion, val_loader, save_path=args.model_home)



