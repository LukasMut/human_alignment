from thingsvision.model_class import Model
from models import CustomModel
from main_eval import evaluate
from typing import List
from utils import jensenshannon
from matplotlib import pyplot as plt

import argparse
import json
import os
import torch
import torchvision

import numpy as np
import pandas as pd


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--data_root",
        type=str,
        help="path/to/things",
        default="/home/space/datasets/things",
    )
    aa("--out_path", type=str, help="path/to/results", default="")
    aa("--model_names", type=str, nargs="+", default=[])
    aa("--module_type_names", type=str, nargs="+", default=["logits"])
    aa(
        "--temperatures",
        type=float,
        nargs="+",
        default=[
            1.0,
            0.75,
            0.5,
            0.25,
            0.1,
            0.075,
            0.05,
            0.025,
            0.01,
            0.0075,
            0.005,
            0.0025,
            0.001,
            0.00075,
            0.0005,
            0.00025,
            0.0001,
            0.000075,
            0.00005,
            0.000025,
            0.00001,
        ],
    )
    aa(
        "--overwrite",
        type=bool,
        help="If set to False, existing dictionary will be updated.",
        default=False,
    )
    aa(
        "--run_models",
        type=bool,
        help="If set to False, probas will be loaded from storage (if possible).",
        default=False,
    )
    aa(
        "--distance",
        type=str,
        default="jensenshannon",
        choices=["cosine", "euclidean", "jensenshannon"],
        help="distance function used for predicting the odd-one-out",
    )
    aa(
        "--ssl_models_path",
        type=str,
        default="/home/space/datasets/things/ssl-models",
        help="Path to converted ssl models from vissl library.",
    )
    aa(
        "--one_hot",
        type=bool,
        help="If set to True, one-hot vectors are used as ground-truth, rather than VICE outputs.",
        default=False,
    )
    aa(
        "--dataset",
        type=str,
        choices=["things", "things-aligned"],
        default="things-aligned",
    )
    args = parser.parse_args()
    return args


def _is_model_name_accepted(name: str):
    name_starts = ["alexnet", "vgg", "res", "vit", "efficient", "clip", "r50"]
    is_ok = any([name.startswith(start) for start in name_starts])
    is_ok &= not name.endswith("_bn")
    is_ok &= not "vit_l" in name
    is_ok &= name == "alexnet" or any(c.isdigit() for c in name)
    return is_ok


def get_logit_module_name(model: Model):
    is_clip = "clip" in model.model_name
    if is_clip:
        module_name = "visual"
    else:
        module_to_iterate = model.model
        module_name = [m[0] for m in module_to_iterate.named_modules()][-1]
    return module_name


def get_penult_module_name(model: Model):
    is_clip = "clip" in model.model_name
    if is_clip:
        module_name = "visual"
    elif model.model_name == "r50-vicreg":
        # This is the only SSL architecure w/o fc layer. For the sake of unity, this assures that penult is avgpool.
        module_name = "avgpool"
    else:
        logit_module_name = get_logit_module_name(model)
        module_name = None
        not_permitted = [
            torch.nn.ReLU,
            torch.nn.GELU,
            torch.nn.Dropout,
        ]
        module_to_iterate = model.model
        for mod_name, mod in module_to_iterate.named_modules():
            is_leaf = not [c for c in mod.children()]
            is_legal = not any([isinstance(mod, cls) for cls in not_permitted])
            is_logit = mod_name == logit_module_name
            if is_leaf and is_legal and not is_logit:
                module_name = mod_name
    return module_name


def get_model_dict(model_names: List[str], dist: str, ssl_models_path: str):
    """Returns a dictionary with logit and penultimate layer module names for every model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = {
        model_name: {
            "logits": {
                "module_name": None,
                "temperature": {dist: None},
            },
            "penultimate": {
                "module_name": None,
                "temperature": {
                    dist: None,
                },
            },
        }
        for model_name in model_names
    }
    for model_name in model_names:
        model = CustomModel(
            model_name=model_name,
            pretrained=True,
            model_path=None,
            device=device,
            backend="pt",
            ssl_models_path=ssl_models_path,
        )
        model_dict[model_name]["logits"]["module_name"] = get_logit_module_name(model)
        model_dict[model_name]["penultimate"]["module_name"] = get_penult_module_name(
            model
        )
    return model_dict


def _get_results_path(out_path: str, temp: float, dist: str, one_hot: bool):
    path = os.path.join(
        out_path, "temperatures", dist + "_" + str(temp) + ("_oh" if one_hot else "")
    )
    return path


def ECE(probas: torch.Tensor, equal_mass: bool = False, n_bins=10):
    """Expected Calibration Error"""
    assert len(probas.shape) == 2
    assert probas.shape[1] == 3

    n = len(probas)
    max_vals, max_idcs = torch.max(probas, dim=1)

    bin_borders = [bin_id / n_bins for bin_id in range(n_bins)]
    if equal_mass:
        bin_borders = torch.quantile(max_vals, torch.tensor(bin_borders))
        bin_borders = sorted(list(set(bin_borders.numpy().tolist())))
        if len(bin_borders) < n_bins:
            n_bins = len(bin_borders)
            print("Reduced number of bins to %d due to proba homogeneity." % n_bins)
    bin_borders += [1.1]

    ece = 0
    sample_counter = 0
    for bin_i, border in enumerate(bin_borders[:-1]):
        vals = max_vals[border <= max_vals]
        idcs = max_idcs[border <= max_vals]
        idcs = idcs[vals < bin_borders[bin_i + 1]]
        vals = vals[vals < bin_borders[bin_i + 1]]
        if len(vals) > 0:
            acc = torch.mean(torch.where(idcs == 0, 1.0, 0.0))
            conf = torch.mean(vals)
            m = len(vals)
            sample_counter += m
            ce = torch.abs(acc - conf)
            ece += m / n * ce
            print(m, "acc:%.3f  conf:%.3f  ce:%f.3f" % (acc, conf, ce))

    assert sample_counter == n
    return ece


def search_temperatures(
    model_dict: dict,
    things_root: str,
    out_path: str,
    temperatures: List[float],
    run_models: bool,
    module_type_names: List[str],
    distance: str,
    one_hot: bool,
    ssl_models_path: str,
    dataset: str,
):
    """Find the temperature scaling with minimal average distance over the VICE-correct triplets and populate the
    dictionary with it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get probas for each configuration
    model_names = [str(k) for k in model_dict.keys()]
    for model_name in model_names:
        for module_type_name in module_type_names:
            for temp in temperatures:
                module_name = model_dict[model_name][module_type_name]["module_name"]
                results_root = _get_results_path(
                    out_path, temp, distance, one_hot=False
                )
                results_path = os.path.join(results_root, model_name, module_name)
                results_exists = os.path.exists(results_path)
                if run_models or not results_exists:
                    # Save a configuration to be loaded in the evaluate function
                    model_dict[model_name][module_type_name]["temperature"][
                        distance
                    ] = temp
                    save_dict(model_dict, out_path, overwrite, one_hot)

                    config = DotDict(
                        {
                            "data_root": things_root,
                            "dataset": dataset,
                            "model_names": [model_name],
                            "module": module_type_name,
                            "distance": distance,
                            "out_path": results_path,
                            "device": device,
                            "batch_size": 8,
                            "num_threads": 4,
                            "ssl_models_path": ssl_models_path,
                            "model_dict_path": get_dict_path(out_path, one_hot),
                        }
                    )
                    print("Evaluating:", model_name, module_name, temp)
                    evaluate(config)
                else:
                    print("Will load probas for:", model_name, module_name, temp)

    probas_vice = None
    if not one_hot:
        # Load vice probas
        print("Loading VICE probas")
        file_modifier = "correct" if dataset == "things-aligned" else "all"
        probas_vice = torch.tensor(
            np.load(
                os.path.join(
                    things_root, "probas", "probabilities_%s_triplets.npy" % file_modifier
                )
            )
        )

    # Load probas for each configuration and select best temperature
    for model_name in model_names:
        for module_type_name in module_type_names:
            min_value = None
            kls = []
            jss = []
            ece = []
            ece_eq_mass = []
            for temp in temperatures:
                module_name = model_dict[model_name][module_type_name]["module_name"]
                results_root = _get_results_path(
                    out_path, temp, distance, one_hot=False
                )
                results_path = os.path.join(results_root, model_name, module_name)

                print("Processing...", model_name, module_name, temp, flush=True)

                with open(os.path.join(results_path, "results.pkl"), "rb") as f:
                    df = pd.read_pickle(f)
                    probas = torch.tensor(df[df.model == model_name]["probas"][0])

                if probas_vice is None:
                    probas_vice = torch.zeros_like(probas)
                    probas_vice[:, 2] = 1

                avg_kl = torch.nn.KLDivLoss()(probas, probas_vice)
                kls.append(avg_kl)

                avg_js = np.mean(
                    [jensenshannon(p, pv) for (p, pv) in zip(probas, probas_vice)]
                )
                jss.append(avg_js)

                ece_val = ECE(probas, equal_mass=False)
                ece_em_val = ECE(probas, equal_mass=True)
                ece.append(ece_val)
                ece_eq_mass.append(ece_em_val)

                print("    js %.4f" % avg_js)
                print("    kl %.4f" % avg_kl)
                print("    ece %.4f" % ece_val)
                print("    eceem %.4f" % ece_em_val)
                if min_value is None or ece_val < min_value:
                    min_value = ece_val
                    model_dict[model_name][module_type_name]["temperature"][
                        distance
                    ] = temp
                    print(f"  New best temp = {temp} (ECE = {ece_val})")

                # Saving the results for all temperatures, for plotting
                scaling_results_folder = os.path.join(out_path, "scaling_results")
                if not os.path.exists(scaling_results_folder):
                    os.makedirs(scaling_results_folder)
                np.save(
                    os.path.join(
                        scaling_results_folder,
                        "_".join(
                            [
                                model_name,
                                module_type_name,
                                distance,
                                str(one_hot),
                                "all_temps",
                            ]
                        ),
                    ),
                    {
                        "temperatures": temperatures,
                        "kls": kls,
                        "jss": kls,
                        "ece": ece,
                        "ece_eq_mass": ece_eq_mass,
                    },
                )


def save_dict(dictionary: dict, out_path: str, overwrite: bool, one_hot: bool):
    os.makedirs(out_path, exist_ok=True)

    if not overwrite:
        try:
            old_dict = load_dict(out_path, one_hot)
            for model_key, model_val in dictionary.items():
                if model_key in old_dict:
                    # If model already in old_dict, update only the new modules
                    for module_key in module_type_names:
                        try:
                            # If module already in old_dict, update only the new distance measure
                            old_dict[model_key][module_key]["temperature"][
                                distance
                            ] = model_val[module_key]["temperature"][distance]
                        except KeyError:
                            # Else, update the whole module entry
                            old_dict[model_key].update(
                                {str(module_key): model_val[module_key]}
                            )
                else:
                    # Else, update the whole model entry
                    old_dict.update({str(model_key): model_val})
            dictionary = old_dict
        except FileNotFoundError:
            print("Could not load dictionary. Creating new one.")

    with open(os.path.join(out_path, get_model_dict_name(one_hot)), "w+") as f:
        json.dump(dictionary, f, indent=4)


def get_dict_path(out_path: str, one_hot: bool):
    return os.path.join(out_path, get_model_dict_name(one_hot))


def load_dict(out_path: str, one_hot: bool):
    with open(get_dict_path(out_path, one_hot), "r") as f:
        dictionary = json.load(f)
    return dictionary


def get_model_dict_name(one_hot: bool):
    name = "model_dict.json"
    if one_hot:
        name = name.replace(".json", "_onehot.json")
    return name


def plot_dist_temp(
    out_path: str,
    model_names: List[str],
    module_type_name: str,
    distance: str,
    one_hot: bool,
):
    scaling_results_folder = os.path.join(out_path, "scaling_results")
    distances = {}
    for model_name in model_names:
        distances[model_name] = np.load(
            os.path.join(
                scaling_results_folder,
                "_".join(
                    [
                        model_name,
                        module_type_name,
                        distance,
                        str(one_hot),
                        "all_temps.npy",
                    ]
                ),
            ),
            allow_pickle=True,
        )[()]
    n_cols = min(len(model_names), 5)
    n_rows = int(np.ceil(len(model_names) / n_cols))

    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows)
    )

    try:
        if len(axs.shape) < 2:
            axs = [axs]
    except AttributeError:
        axs = [[axs]]

    for i_r in range(n_rows):
        for i_c in range(n_cols):
            model_id = n_cols * i_r + i_c
            if model_id >= len(model_names):
                break
            data_for_model = distances[model_names[model_id]]
            ax = axs[i_r][i_c]

            legend = ["KL"]
            try:
                ax.plot(data_for_model["temperatures"], data_for_model["kls"])
                ax.plot(data_for_model["temperatures"], data_for_model["jss"])
                legend.append("JS")
            except KeyError:
                # Backward compatibility
                ax.plot(data_for_model["temperatures"], data_for_model["dists"])

            ax.plot(data_for_model["temperatures"], data_for_model["ece"])
            legend.append("ECE")
            ax.plot(data_for_model["temperatures"], data_for_model["ece_eq_mass"])
            legend.append("ECE_EM")

            ax.set_xscale("log")
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Selection Criterion")
            ax.set_title(model_names[model_id] + " (%s)" % module_type_name)
            ax.legend(legend)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parseargs()

    distance = args.distance
    data_root = args.data_root
    out_path = args.out_path
    temperatures = args.temperatures
    run_models = args.run_models
    module_type_names = args.module_type_names
    args_model_names = args.model_names
    overwrite = args.overwrite
    ssl_models_path = args.ssl_models_path
    one_hot = args.one_hot
    dataset = args.dataset

    model_names = [
        name for name in dir(torchvision.models) if _is_model_name_accepted(name)
    ] + [
        "clip-ViT",
        "clip-RN",
        "r50-simclr",
        "r50-mocov2",
        "r50-jigsaw",
        "r50-colorization",
        "r50-rotnet",
        "r50-swav",
        "r50-vicreg",
        "r50-barlowtwins",
        "vit_base_patch16_224",
        "vit_base_patch32_224",
        "vit_large_patch16_224",
        "vit_small_patch16_224",
        "vit_small_patch32_224",
        "vit_tiny_patch16_224",
        "resnet26",
        'resnetv2_50',
        'resnetv2_101',
        "convnext_femto",
        "convnext_pico",
        "convnext_nano",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ]
    if args_model_names:
        model_names = [
            name
            for name in model_names
            if any([name.startswith(args_name) for args_name in args_model_names])
        ]
    print("Models to process:", model_names)

    model_dict = get_model_dict(
        model_names, dist=distance, ssl_models_path=ssl_models_path
    )

    search_temperatures(
        model_dict,
        data_root,
        out_path,
        temperatures,
        run_models,
        module_type_names,
        distance,
        one_hot,
        ssl_models_path,
        dataset,
    )

    save_dict(model_dict, out_path, overwrite, one_hot)
    print("Done.")
