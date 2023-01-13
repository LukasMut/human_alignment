import argparse
import os
import pickle
import copy
import warnings
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, make_scorer
from thingsvision import get_extractor
from torchvision.datasets import CIFAR100, DTD
import torch.nn.functional as F
from functools import partial

from multiprocessing import get_context, Pool

from tqdm import tqdm
import utils
from main_model_sim_eval import get_module_names

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="things")
    aa(
        "--model_names",
        type=str,
        nargs="+",
        help="models for which we want to extract featues",
    )
    aa(
        "--module",
        type=str,
        choices=["logits", "penultimate"],
        help="module for which to extract features",
    )
    aa("--overall_source", type=str, default="thingsvision")
    aa(
        "--sources",
        type=str,
        nargs="+",
        choices=[
            "custom",
            "torchvision",
            "ssl",
            "google",
            "loss",
            "imagenet",
            "torchvision",
            "vit_same",
            "vit_best",
        ],
        help="Source of (pretrained) models",
    )
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json",
    )
    aa(
        "--n_shot",
        type=int,
        nargs="+",
        help="Number samples per class for training",
        default=10,
    )
    aa(
        "--n_test",
        type=int,
        help="Number samples per class for testing",
        default=100,
    )
    aa(
        "--n_reps",
        type=int,
        help="Number of repetitions per experiment",
        default=1,
    )
    aa(
        "--input_dim",
        type=int,
        help="Side-length of the input images.",
        default=32,
    )
    aa(
        "--regressor_type",
            type=str,
            nargs="+",
            choices=[
                "ridge",
                "knn"
            ],
            help="Few shot model.",
    )
    aa(
        "--n_classes",
        type=int,
        help="Number of classes",
    )
    aa("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    aa(
        "--things_embeddings_path",
        type=str,
        default="/home/space/datasets/things/embeddings/model_features_per_source.pkl",
        help="path/to/things/embeddings/file",
    )
    aa(
        "--transforms_root",
        type=str,
        default="/home/space/datasets/things",
        help="path/to/embeddings",
    )
    aa(
        "--transform_type",
        type=str,
        default="without_norm",
        choices=["without_norm", "with_norm"],
        help="type of transformation matrix being used",
    )
    aa("--out_dir", type=str, help="directory to save the results to")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def train_regression(train_targets: Array, train_features: Array, k: int = None):
    n_train = train_features.shape[0]
    print("N. train:", n_train)

    reg = LogisticRegressionCV(
        Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
        fit_intercept=True,
        penalty="l2",
        # scoring=make_scorer(accuracy_score),
        cv=k,
        max_iter=500,
        solver="sag",
    )

    reg.fit(train_features, train_targets)

    return reg


def train_knn(train_targets: Array, train_features: Array, k: int = 1):
    n_train = train_features.shape[0]
    print("N. train:", n_train)

    reg = KNeighborsRegressor(
        n_neighbors=k,
        algorithm="ball_tree",
    )

    reg.fit(train_features, train_targets)

    return reg


def test_regression(
    regressor,
    test_targets: Array,
    test_features: Array,
):
    n_test = test_features.shape[0]
    print("N. test:", n_test)

    preds = regressor.predict(test_features)
    acc = np.sum([p == t for p, t in zip(preds, test_targets)]) / len(preds)
    try:
        regularization_strength = regressor.C_
        print("Accuracy: %.3f, Regularization:" % acc, regularization_strength)
    except AttributeError:
        print("Accuracy: %.3f" % acc)

    return acc, preds


def regress(
    train_targets: Array,
    train_features: Array,
    test_targets: Array,
    test_features: Array,
    k: int = None,
    regressor: str = "ridge",
):
    if regressor == "ridge":
        reg = train_regression(train_targets, train_features, k)
    if regressor == "knn":
        reg = train_knn(train_targets, train_features)
    else:
        raise ValueError(f"Unknown regressor: {regressor}")
    acc, preds = test_regression(reg, test_targets, test_features)
    return acc, preds


def load_dataset(name: str, data_dir: str, train: bool, transform=None):
    if name == "cifar100":
        dataset = CIFAR100(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "DTD":
        dataset = DTD(
            root=data_dir,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    else:
        raise ValueError("\nUnknown dataset\n")
    return dataset


def get_features_targets(
    class_ids,
    model_name,
    model_params,
    source,
    module,
    data_cfg,
    batch_size,
    train,
    embeddings_root=None,
    ids_subset=None,
    n_batches=1,
    shuffle=False,
    device: str = "cpu",
):
    ids_subset = class_ids if ids_subset is None else ids_subset

    if embeddings_root:
        embeddings = utils.evaluation.load_embeddings(
                embeddings_root=embeddings_root,
                module="embeddings" if module == "penultimate" else "logits"
        )
    else:
        extractor = get_extractor(
            model_name=model_name,
            source=source,
            device=device,
            pretrained=True,
            model_parameters=model_params,
        )
    dataset = load_dataset(
        name=data_cfg.name,
        data_dir=data_cfg.root,
        train=train,
        transform=extractor.get_transformations(),
    )
    features_all = []
    Y_all = []
    for i_batch in range(n_batches):
        X = None
        Y = None
        for i_cls_id, cls_id in enumerate(class_ids):
            if cls_id not in ids_subset:
                continue
            try:
                subset_indices = [
                    i_cls for i_cls, cls in enumerate(dataset.targets) if cls == cls_id
                ]
            except AttributeError:
                subset_indices = [
                    i_cls for i_cls, cls in enumerate(dataset._labels) if cls == cls_id
                ]
            subset = torch.utils.data.Subset(
                dataset,
                subset_indices,
            )
            batches = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=1,
                worker_init_fn=lambda id: np.random.seed(id + i_batch * 4),
            )

            for x, y in batches:
                if X is None:
                    X = [x.to(device)]
                    Y = [i_cls_id] * len(y)
                else:
                    X += [x.to(device)]
                    Y += [i_cls_id] * len(y)
                break
        Y = np.array(Y)

        if (
            source == "torchvision"
            and module in ["penultimate", "encoder.ln"]
            and model_name.startswith("vit")
        ):
            features = extractor.extract_features(
                batches=X,
                module_name=module,
                flatten_acts=False,
            )
            features = features[:, 0]  # select classifier token
            features = features.reshape((features.shape[0], -1))
        else:
            features = extractor.extract_features(
                batches=X,
                module_name=module,
                flatten_acts=True,
            )

        features_all.append(features)
        Y_all.append(Y)
    return features_all, Y_all


def create_config_dicts(args) -> Tuple[FrozenDict, FrozenDict]:
    """Create data and model config dictionaries."""
    model_config = utils.evaluation.load_model_config(args.model_dict_path)
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    model_cfg.names = args.model_names
    model_cfg.modules = get_module_names(model_config, model_cfg.names, args.module)
    model_cfg.module_type = args.module
    model_cfg.sources = args.sources
    model_cfg.input_dim = args.input_dim
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg.name = args.dataset
    data_cfg.category = None
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def run(
    n_shot: int,
    n_test: int,
    n_reps: int,
    class_id_sets: List,
    device: str,
    model_cfg: FrozenDict,
    data_cfg: FrozenDict,
    features_things: Array,
    transforms: Array,
    transform_type: str = None,
    regressor_type: str = "ridge",
):
    transform_options = [False, True]

    results = []
    for class_id_set in class_id_sets:
        for model_name, module, source in zip(
            model_cfg.names, model_cfg.modules, model_cfg.sources
        ):
            # Resolve family name
            if model_name.startswith("OpenCLIP"):
                name, variant, data = model_name.split("_")
                model_params = dict(variant=variant, dataset=data)
            elif model_name.startswith("clip"):
                name, variant = model_name.split("_")
                model_params = dict(variant=variant)
            else:
                name = model_name
                model_params = None
            family_name = utils.analyses.get_family_name(model_name)

            # Extract train features
            train_features_original_all, train_targets_all = get_features_targets(
                class_id_set,
                name,
                model_params,
                source,
                module,
                data_cfg,
                n_shot,
                train=True,
                n_batches=n_reps,
                shuffle=True,
                device=device,
            )

            things_mean = np.mean(
                features_things[source][model_name][model_cfg.module_type],
                # axis=0,
            )
            things_std = np.std(
                features_things[source][model_name][model_cfg.module_type],
                # axis=0,
            )

            # Train regression w and w/o transform
            regressors = {to: [] for to in transform_options}
            for (train_features_original, train_targets) in zip(
                train_features_original_all, train_targets_all
            ):
                for use_transforms in transform_options:
                    if use_transforms:
                        try:
                            transform = transforms[source][model_name][
                                model_cfg.module_type
                            ]
                        except KeyError:
                            warnings.warn(
                                message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                                category=UserWarning,
                            )
                            continue

                        train_features = (
                            train_features_original - things_mean
                        ) / things_std
                        # train_features = train_features_original
                        train_features = train_features @ transform
                        if transform_type == "with_norm":
                            train_features = torch.from_numpy(train_features)
                            train_features = (
                                F.normalize(train_features, dim=1).cpu().numpy()
                            )

                    else:
                        train_features = train_features_original - things_mean

                    if regressor_type == "ridge":
                        regressor = train_regression(
                            train_targets, train_features, k=n_shot
                        )
                    elif regressor_type == "knn":
                        regressor = train_knn(train_targets, train_features)
                    else:
                        raise ValueError(f"Unknown regressor: {args.regressor}")
                    regressors[use_transforms].append(regressor)

            # Extract and evaluate features w and w/o transform. Due to memory constraints, for each class individually.
            for i_rep in range(n_reps):
                accuracies = {a: [] for a in transform_options}
                for cls_id in [class_id_set]:
                    test_features_original, test_targets = get_features_targets(
                        class_id_set,
                        name,
                        model_params,
                        source,
                        module,
                        data_cfg,
                        n_test,
                        train=False,
                        shuffle=False,
                        ids_subset=cls_id if type(cls_id) == list else [cls_id],
                        device=device,
                    )
                    test_features_original = test_features_original[0]
                    test_targets = test_targets[0]

                    for use_transforms in transform_options:
                        if use_transforms:
                            try:
                                transform = transforms[source][model_name][
                                    model_cfg.module_type
                                ]
                            except KeyError:
                                warnings.warn(
                                    message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                                    category=UserWarning,
                                )
                                continue

                            test_features = (
                                test_features_original - things_mean
                            ) / things_std
                            # test_features = test_features_original
                            test_features = test_features @ transform
                            if transform_type == "with_norm":
                                test_features = torch.from_numpy(test_features)
                                test_features = (
                                    F.normalize(test_features, dim=1).cpu().numpy()
                                )
                        else:
                            test_features = test_features_original - things_mean

                        acc, pred = test_regression(
                            regressors[use_transforms][i_rep],
                            test_targets,
                            test_features,
                        )
                        accuracies[use_transforms].append(acc)

                # Store results for all classes
                for use_transforms in transform_options:
                    summary = {
                        "accuracy": np.mean(accuracies[use_transforms]),
                        "model": model_name,
                        "module": model_cfg.module_type,
                        "source": source,
                        "family": family_name,
                        "dataset": data_cfg.name,
                        "transform": use_transforms,
                        "classes": class_id_set,
                        "n_train": n_shot,
                        "repetition": i_rep,
                        "transform_type": transform_type if use_transforms else None,
                        "regressor": regressor_type,
                    }
                    results.append(summary)

            print(summary)

    results = pd.DataFrame(results)
    return results


def _add_model(model_cfg, f):
    return f(model_cfg=model_cfg)


if __name__ == "__main__":
    start_t = datetime.now()

    # parse arguments
    args = parseargs()
    class_id_sets = [[i for i in range(args.n_classes)]]
    n_shot = args.n_shot
    n_test = args.n_test
    device = torch.device(args.device)
    model_cfg, data_cfg = create_config_dicts(args)

    # Load transforms and things embeddings
    features_things = utils.evaluation.load_features(path=args.things_embeddings_path)
    transforms = utils.evaluation.helpers.load_transforms(
        root=args.transforms_root, type=args.transform_type
    )

    # Reduce to the needed models
    transforms = {
        src: {
            mdl: transforms[src][mdl]
            for mdl_i, mdl in enumerate(args.model_names)
            if args.sources[mdl_i] == src
        }
        for src in args.sources
    }

    features_things = {
        src: {
            mdl: features_things[src][mdl]
            for mdl_i, mdl in enumerate(args.model_names)
            if args.sources[mdl_i] == src
        }
        for src in args.sources
    }

    # Do few-shot
    all_results = []
    regressor_types = args.regressor_type
    n_shots = args.n_shot
    for regressor_type in regressor_types:
        for shots in n_shots:
            if regressor_type == "ridge" and shots==1:
                continue
            args.n_shot = shots
            args.regressor_type = regressor_type
            model_cfg, data_cfg = create_config_dicts(args)

            if False:#args.device == "cpu":
                model_names = args.model_names
                sources = args.sources
                model_cfgs = []
                for mod, src in zip(model_names, sources):
                    mod_args = copy.copy(args)
                    mod_args.model_name = mod
                    mod_args.source = src
                    model_cfgs.append(create_config_dicts(mod_args)[0])

                # args.n_shot = shots
                _, data_cfg = create_config_dicts(args)
                runp = partial(
                    run,
                    n_shot=args.n_shot,
                    n_test=args.n_test,
                    n_reps=args.n_reps,
                    class_id_sets=class_id_sets,
                    device=args.device,
                    data_cfg=data_cfg,
                    features_things=features_things,
                    transforms=transforms,
                    transform_type=args.transform_type,
                    regressor_type=args.regressor_type,
                )
                mapp = partial(
                    _add_model, f=runp
                )

                with get_context("spawn").Pool() as pool:
                    all_results = list(
                        tqdm(pool.map(mapp, model_cfgs), total=len(model_cfgs)),
                        chunksize=10,
                    )
            else:
                results = run(
                    n_shot=args.n_shot,
                    n_test=args.n_test,
                    n_reps=args.n_reps,
                    class_id_sets=class_id_sets,
                    device=args.device,
                    model_cfg=model_cfg,
                    data_cfg=data_cfg,
                    features_things=features_things,
                    transforms=transforms,
                    transform_type=args.transform_type,
                    regressor_type=args.regressor_type,
                )
            all_results.append(results)
    results = pd.concat(all_results)

    out_path = os.path.join(
        args.out_dir, args.dataset, args.overall_source, args.module
    )
    if not os.path.exists(out_path):
        print("\nOutput directory does not exist...")
        print("Creating output directory to save results...\n")
        os.makedirs(out_path)

    results.to_pickle(os.path.join(out_path, "fewshot_results.pkl"))

    print("Elapsed time (init):", datetime.now() - start_t)
