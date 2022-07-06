from thingsvision.model_class import Model
from models import CustomModel
from main_eval import evaluate
from typing import List

import argparse
import json
import os
import torch
import torchvision

import numpy as np


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
            0.5,
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
            0.0005,
            0.0001,
            0.00005,
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
        help="Path to converted ssl models from vissl library."
    )
    args = parser.parse_args()
    return args


def _is_model_name_accepted(name: str):
    name_starts = ["alexnet", "vgg", "res", "vit", "efficient", "clip"]
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
            model_name=model_name, pretrained=True, model_path=None, device=device, backend="pt",
            ssl_models_path=ssl_models_path
        )
        model_dict[model_name]["logits"]["module_name"] = get_logit_module_name(model)
        model_dict[model_name]["penultimate"]["module_name"] = get_penult_module_name(
            model
        )
    return model_dict


def _get_results_path(out_path: str, temp: float, dist: str):
    path = os.path.join(out_path, dist + "_" + str(temp))
    return path


def seach_temperatures(
    model_dict: dict,
    things_root: str,
    out_path: str,
    temperatures: List[float],
    run_models: bool,
    module_type_names: List[str],
    distance: str,
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
                results_root = _get_results_path(out_path, temp, distance)
                results_exists = os.path.exists(
                    os.path.join(results_root, model_name, module_name)
                )
                if run_models or not results_exists:
                    # Save a configuration to be loaded in the evaluate function
                    model_dict[model_name][module_type_name]["temperature"][
                        distance
                    ] = temp
                    save_dict(model_dict, out_path, overwrite)

                    config = DotDict(
                        {
                            "data_root": things_root,
                            "dataset": "things-aligned",
                            "model_names": [model_name],
                            "module": module_type_name,
                            "distance": distance,
                            "out_path": results_root,
                            "device": device,
                            "batch_size": 8,
                            "num_threads": 4,
                        }
                    )
                    print("Evaluating:", model_name, module_name, temp)
                    evaluate(config)
                else:
                    print("Will load probas for:", model_name, module_name, temp)

    # Load vice probas
    print("Loading VICE probas")
    probas_vice = np.load(
        os.path.join(things_root, "probas", "probabilities_correct_triplets.npy")
    )

    # Load probas for each configuration and select best temperature
    for model_name in model_names:
        for module_type_name in module_type_names:
            min_js = None
            for temp in temperatures:
                module_name = model_dict[model_name][module_type_name]["module_name"]
                results_root = _get_results_path(out_path, temp, distance)
                results_path = os.path.join(results_root, model_name, module_name)

                print("Processing...", model_name, module_name, temp, flush=True)

                probas = np.load(os.path.join(results_path, "triplet_probas.npy"))

                avg_dist = torch.nn.CrossEntropyLoss()(
                    torch.tensor(probas), torch.tensor(probas_vice)
                )

                if min_js is None or avg_dist < min_js:
                    min_js = avg_dist
                    model_dict[model_name][module_type_name]["temperature"][
                        distance
                    ] = temp
                    print(f"  New best temp = {temp} (dist. = {avg_dist})")


def save_dict(dictionary: dict, out_path: str, overwrite: bool):
    if not overwrite:
        try:
            old_dict = load_dict(out_path)
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

    with open(os.path.join(out_path, "model_dict.json"), "w+") as f:
        json.dump(dictionary, f, indent=4)


def load_dict(out_path: str):
    with open(os.path.join(out_path, "model_dict.json"), "r") as f:
        dictionary = json.load(f)
    return dictionary


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

    model_names = [
        name for name in dir(torchvision.models) if _is_model_name_accepted(name)
    ] + ["clip-ViT", "clip-RN"]
    if args_model_names:
        model_names = [
            name
            for name in model_names
            if any([name.startswith(args_name) for args_name in args_model_names])
        ]
    print("Models to process:", model_names)

    model_dict = get_model_dict(model_names, dist=distance, ssl_models_path=ssl_models_path)

    seach_temperatures(
        model_dict,
        data_root,
        out_path,
        temperatures,
        run_models,
        module_type_names,
        distance,
    )

    save_dict(model_dict, out_path, overwrite)
    print("Done.")
