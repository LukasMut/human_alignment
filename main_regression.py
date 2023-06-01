import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

import utils


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds in k-fold cross-validation.",
    )
    aa("--model_names", type=str, nargs="+", default=[])
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    aa("--load", action="store_true", help="Load results if they exist.")
    args = parser.parse_args()
    return args


Array = np.ndarray


def regress(
    train_target_features: Array,
    train_source_features: Array,
    test_target_features: Array,
    test_source_features: Array,
    k: int = None,
):
    train_target_features = train_target_features.T
    test_target_features = test_target_features.T

    n_dimensions = len(train_target_features)
    n_test = test_target_features.shape[1]

    reg = RidgeCV(
        alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
        fit_intercept=True,
        scoring=None,
        cv=k,
    )

    r2 = np.zeros([n_dimensions])
    preds = np.zeros([n_dimensions, n_test])
    for d in range(n_dimensions):
        target = train_target_features[d, :, None]
        source = train_source_features
        reg.fit(source, target)

        target = test_target_features[d, :, None]
        source = test_source_features
        score = reg.score(source, target)
        preds[d] = reg.predict(source)[:, 0]
        r2[d] = score
        alpha = reg.alpha_
        print("   ", score, alpha)

    return r2, preds


def regress_k_fold(
    target_features: Array, source_features: Array, k: int, rnd_seed: int
):
    n_objects = target_features.shape[0]
    n_dimensions = target_features.shape[1]
    r2s = np.zeros([n_dimensions, 5])
    preds = np.zeros([n_dimensions, source_features.shape[0]])
    truths = np.zeros([n_dimensions, source_features.shape[0]])
    idcs = np.zeros([source_features.shape[0]])

    objects = np.arange(n_objects)
    kf = KFold(n_splits=k, random_state=rnd_seed, shuffle=True)

    sample_cnt = 0
    for k_i, (train_idx, test_idx) in enumerate(kf.split(objects)):
        train_target_features = target_features[train_idx]
        test_target_features = target_features[test_idx]
        train_source_features = source_features[train_idx]
        test_source_features = source_features[test_idx]
        r2, pred = regress(
            train_target_features=train_target_features,
            train_source_features=train_source_features,
            test_target_features=test_target_features,
            test_source_features=test_source_features,
        )

        r2s[:, k_i] = r2
        fold_size = len(test_idx)
        idcs[sample_cnt : (sample_cnt + fold_size)] = test_idx
        preds[:, sample_cnt : (sample_cnt + fold_size)] = pred
        truths[:, sample_cnt : (sample_cnt + fold_size)] = test_target_features.T
        sample_cnt += fold_size
    return r2s, preds, truths, idcs.astype(int)


def triplet_task(features: Array, data_root: str, k: int, rnd_seed: int):
    n_objects = features.shape[0]
    objects = np.arange(n_objects)
    triplets = utils.probing.load_triplets(data_root)

    kf = KFold(n_splits=k, random_state=rnd_seed, shuffle=True)
    accs = np.zeros([k])

    for k_i, (train_idx, test_idx) in enumerate(kf.split(objects)):
        train_objects = objects[train_idx]
        triplet_partitioning = utils.probing.partition_triplets(
            triplets=triplets,
            train_objects=train_objects,
        )

        choices, _ = utils.evaluation.get_predictions(
            features, np.array(triplet_partitioning["val"])
        )
        acc = utils.evaluation.accuracy(choices)
        accs[k_i] = acc

    return accs


if __name__ == "__main__":
    # parse arguments
    args = parseargs()

    k = args.n_folds
    rnd_seed = args.rnd_seed

    dataset_path = args.data_root
    vice_path = os.path.join(dataset_path, "dimensions/vice_embedding.npy")
    features_path = os.path.join(
        dataset_path, "embeddings/model_features_per_source.pkl"
    )
    out_path = os.path.join(dataset_path, "regression")
    out_file_path = os.path.join(dataset_path, "regression_results.pkl")
    if not os.path.exists(out_path):
        print("\nOutput directory does not exist...")
        print("Creating output directory to save results...\n")
        os.makedirs(out_path)

    # Load vice embeddings
    vice_features = np.load(vice_path)
    n_features = vice_features.shape[1]

    # Load object embeddings for all models
    with open(features_path, "rb") as f:
        features_src = pd.read_pickle(f)

    features = {}
    sources = {}
    for s in features_src.keys():
        if s != "vit_best":
            features.update(features_src[s])
            sources.update({str(k): str(s) for k in features_src[s].keys()})

    # Filter models if necessary
    model_names = [str(m) for m in features.keys()]
    if args.model_names:
        model_names = [m for m in model_names if m in args.model_names]

    print("Models to run:", model_names)

    # Run regression
    for m_i, model in enumerate(model_names):
        out_file_path = os.path.join(
            out_path, "regression_results_k%d_%s.pkl" % (k, model.replace("/", ""))
        )
        results = {model: {}}
        load = args.load
        if load:
            try:
                with open(out_file_path, "rb") as f:
                    results = pd.read_pickle(f)
                print("  Loaded.")
            except FileNotFoundError:
                load = False

        # Regress on targets
        for layer in features[model].keys():
            print(
                "(%d/%d)" % (m_i + 1, len(model_names)),
                model,
                layer,
                "dim=%d" % features[model][layer].shape[1],
                flush=True,
            )
            if not load:
                r2s, preds, truths, idcs = regress_k_fold(
                    target_features=vice_features,
                    source_features=features[model][layer],
                    k=k,
                    rnd_seed=rnd_seed,
                )

                index_inverse = [a[1] for a in sorted(zip(idcs, np.arange(len(idcs))))]

                preds = preds[:, index_inverse]
                truth = truths[:, index_inverse]

                results[model][layer] = {
                    "r2_per_fold": r2s,
                    "r2": np.mean(r2s, axis=1),
                    "predictions": preds,
                    "targets": truths,
                }

        # Save intermediate regression results
        if not load:
            pd.DataFrame(results).to_pickle(out_file_path)

        # Do triplet task
        for layer in features[model].keys():
            accs = triplet_task(
                features=results[model][layer]["predictions"].T,
                data_root=dataset_path,
                k=k,
                rnd_seed=rnd_seed,
            )

            results[model][layer].update(
                {
                    "accuracy": np.mean(accs),
                    "accuracy_per_fold": accs,
                }
            )

        # Save triplet-task regression results
        pd.DataFrame(results).to_pickle(out_file_path)
