import argparse
import json
import os

from torch_geometric.loader import DataLoader
from learning.data_models import StreamInstanceClassifierInfo, StreamInstanceClassifierV2Info
from learning.gnn.data import (
    DeviceAwareLoaderWrapper,
    EvaluationDatasetSampler,
    PLOIAblationDataset,
    TrainingDatasetSampler,
    construct_input,
    HyperModelInfo,
    TrainingDataset,
    Dataset,
    construct_hypermodel_input_faster,
    construct_stream_classifier_input_v2,
    construct_with_problem_graph,
    get_base_datapath,
)
from learning.gnn.models import HyperClassifier, PLOIAblationModel, StreamInstanceClassifier, StreamInstanceClassifierV2
from learning.gnn.train import evaluate_model_loss, evaluate_model_stream, train_model_graphnetwork
from functools import partial
import torch


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-home",
        type=str,
        default="/tmp",
        help = "A directory in which logs and parameters will be saved during training"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help = "If you only want to test a model (not end-to-end)"
    )
    parser.add_argument(
        "--model", type=str, choices=["hyper", "streamclass", "streamclassv2", "ploiablation"], default="hyper",
        help = "The type of model you want to train. See learning/gnn/models.py for more information"
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help = "The number of epochs to train for"
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help = "The number of epochs between tests on the validation set and saving model parameters"
    )
    parser.add_argument(
        "--pos-weight", type=float, default=3.0,
        help="The weighting given to positive examples in the training set"
    )
    parser.add_argument(
        "--stratify-train-prop", type=float, default=None,
        help = "The proportion of positive examples shown to the model" #TODO: is this correct?
    )
    parser.add_argument("--lr", default=0.0001, type=float, help = "The learning rate")
    parser.add_argument("--gradient-batch-size", default=1, type=int, help = "") #TODO: what is this
    parser.add_argument("--batch-size", default=128, type=int, help = "The batch size")
    parser.add_argument("--num-preprocessors", default=8, type=int, help = "The number of cpu cores allowed to help preprocess the data")
    parser.add_argument("--from-best", action="store_true", help = "Do you want to use the best.pt file saved in the model directory during testing")
    parser.add_argument("--use-problem-graph", action="store_true", help = "Do you want to use the problem graph model")
    parser.add_argument(
        "--datafile",
        type=str,
        help="A path to a json file containing train and validation keys wich have a list of pkl paths as values.",
    )
    parser.add_argument("--debug", action="store_true", help = "Are you debugging?")
    parser.add_argument("--epoch-size", default=1280, type=int, help = "The number of labels shown per epoch")
    parser.add_argument("--preprocess-all", action="store_true", help = "Do you want to preprocess all of the data, or processes it when it is needed?")
    parser.add_argument("--ablation", action="store_true", help = "Are you doing an ablation study?") # what exactly does this do?
    parser.add_argument("--feature-size", type=int, default = 16) 
    parser.add_argument("--hidden-size", type=int, default = 16)    
    parser.add_argument("--decrease-score-with-depth", action="store_true") 
    parser.add_argument("--score-initial-objects", action="store_true")    
    parser.add_argument("--trainset-prop", type=float, default = 1)    
    return parser


if __name__ == "__main__":
    args = make_argument_parser().parse_args()
    base_datapath = get_base_datapath()
    with open(args.datafile, "r") as f:
        data = json.load(f)
    train_files = [os.path.join(base_datapath, d) if not d.startswith('/') else d for d in data["train"]]
    if args.trainset_prop < 1:
        train_files = train_files[:int(len(train_files) * args.trainset_prop)]
    val_files = [os.path.join(base_datapath, d) if not d.startswith('/') else d for d in data["validation"]]
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of validation files: {len(val_files)}")

    if args.debug:
        train_files = train_files[:1]
        val_files = train_files[:1]

    if not os.path.exists(args.model_home):
        os.makedirs(args.model_home, exist_ok=True)

    if not args.test_only:
        with open(os.path.join(args.model_home, "hyperparameters.txt"), "w") as f:
            f.write(f"Model {args.model}\n")
            f.write(f"Epochs {args.epochs}\n")
            f.write(f"Pos weight {args.pos_weight}\n")
            f.write(f"Stratify train prop {args.stratify_train_prop}\n")
            f.write(f"learning rate {args.lr}\n")
            f.write(f"Gadient batch size {args.gradient_batch_size}\n")
            f.write(f"Batch size {args.batch_size}\n")
            f.write(f"Use problem graph {args.use_problem_graph}\n")
            f.write(f"Epoch size {args.epoch_size}\n")
            f.write(f"Ablation {args.ablation}\n")
            if args.model == "streamclassv2":
                f.write(f"Feature size {args.feature_size}\n")
                f.write(f"Hidden size {args.hidden_size}\n")
                f.write(f"Score Initial {args.score_initial_objects}\n")
                f.write(f"Decrease With Depth {args.decrease_score_with_depth}\n")

    evaluate_model = evaluate_model_stream
    if args.model == "hyper":
        input_fn = construct_hypermodel_input_faster
        if args.use_problem_graph:
            input_fn = construct_with_problem_graph(input_fn)
        model_info_class = HyperModelInfo
        # TODO: decide what we want to do about this, using problem graph currently requires GNNS
        if args.ablation:
            assert not args.use_problem_graph, "Can't use problem graph without gnns"
        model_fn = lambda model_info: HyperClassifier(
            model_info,
            with_problem_graph=args.use_problem_graph,
            use_gnns=not args.ablation,
        )
    elif args.model == "streamclass":
        input_fn = construct_input
        model_info_class = StreamInstanceClassifierInfo
        model_fn = lambda model_info: StreamInstanceClassifier(
            model_info, use_gcn=True, use_object_model=False
        )
    elif args.model == "streamclassv2":
        assert args.batch_size == 1, "Batching not yet supported for StreamInstanceClassifierV2Info"
        input_fn = construct_stream_classifier_input_v2
        model_info_class = StreamInstanceClassifierV2Info
        model_fn = lambda model_info: StreamInstanceClassifierV2(
            model_info,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            decrease_score_with_depth=args.decrease_score_with_depth,
            score_initial_objects=args.score_initial_objects
        )
    elif args.model == "ploiablation":
        TrainingDataset = PLOIAblationDataset
        Dataset = PLOIAblationDataset
        TrainingDatasetSampler = lambda *args, **kwargs: None
        EvaluationDatasetSampler = lambda *args, **kwargs: None
        evaluate_model = evaluate_model_loss
        input_fn = None
        model_info_class = StreamInstanceClassifierV2Info
        model_fn = lambda model_info: PLOIAblationModel(model_info, feature_size = args.feature_size, hidden_size = args.hidden_size)
    else:
        raise ValueError


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    valset = Dataset(
        input_fn,
        model_info_class,
        preprocess_all=args.preprocess_all,
        clear_memory=False
    )
    valset.from_pkl_files(*val_files)
    valset.prepare()
    val_sampler = EvaluationDatasetSampler(valset)
    val_loader = DataLoader(
        valset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_preprocessors,
    )

    model = model_fn(valset.model_info)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=args.pos_weight * torch.ones([1]))

    criterion.to(device)
    model.to(device)

    if args.from_best:
        model.load_state_dict(torch.load(os.path.join(args.model_home, "best.pt")))

    if not args.test_only:
        trainset = TrainingDataset(
            input_fn,
            model_info_class,
            preprocess_all=args.preprocess_all,
        )
        trainset.from_pkl_files(*train_files)
        trainset.prepare()
        train_sampler = TrainingDatasetSampler(
            trainset, epoch_size=args.epoch_size, stratify_prop=args.stratify_train_prop
        )
        train_loader = DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_preprocessors,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model_graphnetwork(
            model,
            dict(
                train=DeviceAwareLoaderWrapper(train_loader, device),
                val=DeviceAwareLoaderWrapper(val_loader, device),
            ),
            criterion=criterion,
            optimizer=optimizer,
            evaluate_model=evaluate_model,
            step_every=args.gradient_batch_size,
            save_every=args.save_every,
            epochs=args.epochs,
            save_folder=args.model_home,
        )
    # Load the best checkoibt for evaluation
    model.load_state_dict(torch.load(os.path.join(args.model_home, "best.pt")))

    evaluate_model(
        model,
        criterion,
        DeviceAwareLoaderWrapper(val_loader, device),
        save_path=args.model_home,
    )
