
import argparse
import json
import os
from learning.data_models import StreamInstanceClassifierInfo
from learning.gnn.data import construct_input, HyperModelInfo, TrainingDataset, Dataset, construct_hypermodel_input, get_base_datapath, get_pddl_key, query_data
from learning.gnn.models import HyperClassifier, StreamInstanceClassifier
from learning.gnn.train import evaluate_model, train_model_graphnetwork
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
    parser.add_argument(
        "--augment-data",
        action="store_true"
    )
    parser.add_argument(
        "--stratify-train-prop",
        type=float,
        default=None
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float
    )
    parser.add_argument(
        "--gradient-batch-size",
        default=10,
        type=int
    )
    parser.add_argument(
        '--from-best',
        action="store_true"
    )
    parser.add_argument(
        '--datafile',
        type=str,
        help="A path to a json file containing train and validation keys wich have a list of pkl paths as values."
    )
    return parser


if __name__ == '__main__':
    args = make_argument_parser().parse_args()
    base_datapath = get_base_datapath()
    with open(args.datafile, 'r') as f:
        data = json.load(f)
    train_files = [os.path.join(base_datapath, d) for d in data['train']]
    val_files = [os.path.join(base_datapath, d) for d in data['validation']]

    if not os.path.exists(args.model_home):
        os.makedirs(args.model_home, exist_ok=True)

    if args.model == 'hyper':
        input_fn = construct_hypermodel_input
        model_info_class = HyperModelInfo
        model_fn = lambda model_info: HyperClassifier(
            node_feature_size=model_info.node_feature_size,
            edge_feature_size=model_info.edge_feature_size,
            stream_domains=model_info.stream_domains[1:],
            stream_num_inputs=model_info.stream_num_inputs[1:],
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

    valset = Dataset(
        input_fn,
        model_info_class,
        preprocess_all=False,
        max_per_run=200
    )
    valset.from_pkl_files(*val_files)
    valset.prepare()

    model = model_fn(valset.model_info)

    if args.from_best:
        model.load_state_dict(torch.load(os.path.join(args.model_home, 'best.pt')))

    if not args.test_only:
        trainset = TrainingDataset(
            input_fn,
            model_info_class,
            augment=args.augment_data,
            stratify_prop=args.stratify_train_prop,
            preprocess_all=False
        )
        trainset.from_pkl_files(*train_files)
        trainset.prepare()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=args.pos_weight*torch.ones([1]))
        train_model_graphnetwork(
            model,
            dict(train=trainset, val=valset),
            criterion=criterion,
            optimizer=optimizer,
            step_every=args.gradient_batch_size,
            save_every=args.save_every,
            epochs=args.epochs,
            save_folder=args.model_home
        )
        # Load the best checkoibt for evaluation
        model.load_state_dict(torch.load(os.path.join(args.model_home, 'best.pt')))
 
    evaluate_model(model, valset, save_path=args.model_home)



