import os
import pickle
import shutil
import sys
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from utils.analyses import Mapper

Array = np.ndarray

KFOLDS = [3, 4]


def load_probing_results(root: str) -> pd.DataFrame:
    """Load linear probing results into memory."""
    return pd.read_pickle(os.path.join(root, "probing_results.pkl"))


def exclude_vit_subset(results: pd.DataFrame, vit_subset: str = "vit_best") -> None:
    """Exclude a subset of ViTs (<vit_same> or <vit_best>) from results dataframe."""
    results.drop(results[results.source == vit_subset].index, axis=0, inplace=True)
    results.reset_index(drop=True, inplace=True)


def add_meta_info(results: pd.DataFrame) -> pd.DataFrame:
    # initialize mapper class
    mapper = Mapper(results)
    # add information about training objective(s) to dataframe
    results["training"] = mapper.get_training_objectives()
    # modify information about architecture
    results["family"] = [
        utils.analyses.get_family_name(model) for model in results.model.values
    ]
    return results


def partition_into_modules(results: pd.DataFrame) -> List[pd.DataFrame]:
    """Partition results into subsets for the penultimate and logits layer respectively."""
    return [results[results.module == module] for module in ["penultimate", "logits"]]


def filter_best_results(probing_results: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    kfold_subset = probing_results[probing_results.n_folds.isin(KFOLDS)]
    best_results = defaultdict(dict)
    for i, row in tqdm(kfold_subset.iterrows(), desc="Entry"):
        # skip entry if cross-entropy error is NaN or Inf
        if (np.isnan(row["cross-entropy"]) or np.isinf(row["cross-entropy"])):
            continue
        # skip entry if probing odd-one-out accuracy is 1.0
        if (row["cross-entropy"] == np.log(3) or row.probing == float(1)):
            continue
        if row.model in best_results:
            # skip entry if probing odd-one-out accuarcy is worse than previous
            if row["cross-entropy"] > best_results[row.model]["cross-entropy"]:
            # if row.probing < best_results[row.model]["probing"]:
                continue
        best_results[row.model]["index"] = i
        best_results[row.model]["probing"] = row.probing
        best_results[row.model]["cross-entropy"] = row["cross-entropy"]
    indices = np.asarray([vals["index"] for vals in best_results.values()])
    best_results = kfold_subset.filter(indices, axis=0)
    # best_results.drop("choices", axis=1, inplace=True)
    return best_results


def join_modules(
    results_logits: pd.DataFrame, results_penultimate: pd.DataFrame
) -> pd.DataFrame:
    return pd.concat([results_logits, results_penultimate], axis=0, ignore_index=True)


def get_best_probing_results(root: str) -> pd.DataFrame:
    # load linear probing results into memory
    probing_results = load_probing_results(root)
    # exclude ViTs from <vit_best> source
    exclude_vit_subset(probing_results)
    # add information about training objective(s) to results dataframe
    probing_results = add_meta_info(probing_results)
    # partition probing results into subsets for the penultimate and logits layer respectively
    penultimate_probing_results, logits_probing_results = partition_into_modules(
        probing_results
    )
    # filter for best hyperparameters
    best_penultimate_probing_results = filter_best_results(penultimate_probing_results)
    best_logits_probing_results = filter_best_results(logits_probing_results)
    # join best results for penultimate and logits layer
    best_probing_results = join_modules(
        best_logits_probing_results, best_penultimate_probing_results
    )
    return best_probing_results


def find_best_transforms(
    root: str, best_probing_results: pd.DataFrame
) -> Dict[str, Dict[str, Dict[str, Array]]]:
    transforms = defaultdict(lambda: defaultdict(dict))
    missing_transforms = 0
    for _, row in tqdm(best_probing_results.iterrows(), desc="Model"):
        source = row.source
        name = row.model
        module = row.module
        subdir = os.path.join(
            root,
            source,
            name,
            module,
            str(row.n_folds),
            str(row.l2_reg),
            row.optim,
            str(row.lr),
        )
        try:
            transform = load_transform(subdir)
            transforms[source][name][module] = transform
        except FileNotFoundError:
            warnings.warn(
                message=f"\nCannot find transformation matrix in subdirectory: {subdir}\nContinuing with next entry in results dataframe...\n",
                category=UserWarning,
            )
            missing_transforms += 1
            continue
        # delete subdirectory for current model
        # shutil.rmtree(os.path.join(root, source, name, module, str(row.n_folds), str(row.l2_reg)))
    print(
        f"\n{missing_transforms} transformation matrices are missing.\nPlease run grid search again for models with missing transformation matrices.\n"
    )
    return transforms


def load_transform(subdir: str) -> Array:
    with open(os.path.join(subdir, "transform.npy"), "rb") as f:
        transform = np.load(f)
    return transform


def save_transforms(
    root: str, transforms: Dict[str, Dict[str, Dict[str, Array]]]
) -> None:
    with open(os.path.join(root, "transforms.pkl"), "wb") as f:
        pickle.dump(transforms, f)


def save_results(root: str, best_probing_results: pd.DataFrame) -> None:
    best_probing_results.drop("n_folds", axis=1, inplace=True)
    best_probing_results.to_pickle(os.path.join(root, "best_probing_results.pkl"))


if __name__ == "__main__":
    root = sys.argv[1]
    best_probing_results = get_best_probing_results(root)
    transforms = find_best_transforms(root, best_probing_results)
    save_transforms(root, dict(transforms))
    save_results(root, best_probing_results)
