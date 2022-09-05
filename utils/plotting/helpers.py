import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots.constants.color import rgb

Array = np.ndarray

PALETTE = {
    "Image/Text": "darkmagenta",
    "Supervised (ImageNet 1k)": "coral",
    "Supervised (ImageNet 21k)": "darkcyan",
    "Supervised (JFT 30k)": "black",
    "Self-Supervised": "darkgreen",
    "VICE": "red",
}

MARKERS = {
    "Image/Text": "o",
    "Supervised (ImageNet 1k)": "s",
    "Supervised (ImageNet 21k)": "P",
    "Supervised (JFT 30k)": "X",
    "Self-Supervised": "D",
    "VICE": "*",
}


def set_context() -> None:
    sns.set_context("paper")


def concat_images(images: Array, top_k: int) -> Array:
    img_combination = np.concatenate(
        [
            np.concatenate([img for img in images[: int(top_k / 2)]], axis=1),
            np.concatenate([img for img in images[int(top_k / 2) :]], axis=1),
        ],
        axis=0,
    )
    return img_combination


def visualize_dimension(
    ax: Any, images: Array, dimension: Array, top_k: int = 6
) -> None:
    # sort dimension by weights in decending order and get top-k objects
    topk_objects = np.argsort(-dimension)[:top_k]
    topk_images = images[topk_objects]
    img_comb = concat_images(images=topk_images, top_k=top_k)
    for spine in ax.spines:
        ax.spines[spine].set_color(rgb.tue_dark)
        ax.spines[spine].set_linewidth(7)
    ax.imshow(img_comb)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()


def plot_conceptwise_accuracies(
    concept_subset: pd.DataFrame,
    ylabel: bool,
    xlabel: bool,
    probing: bool,
) -> None:
    # sort models by their odd-one-out accuracy in descending order
    ymin = 0.1
    ymax = 0.9
    concept_subset.sort_values(
        by=["odd-one-out-accuracy"], axis=0, ascending=False, inplace=True
    )
    set_context()
    ax = sns.swarmplot(
        data=concept_subset,
        x="family",
        y="odd-one-out-accuracy",
        hue="training",
        orient="v",
        edgecolor="gray",
        s=14,
        alpha=0.7,
        palette=PALETTE,
    )
    ax.set_ylim([ymin, ymax])
    if xlabel:
        ax.set_xticklabels(
            labels=concept_subset.family.unique(), fontsize=34, rotation=40, ha="right"
        )
    else:
        ax.set_xticks([])
    if ylabel:
        label = (
            "Odd-one-out accuracy after probing"
            if probing
            else "Zero-shot odd-one-out accuracy"
        )
        ax.set_ylabel(label, fontsize=35, labelpad=20)
        ax.set_yticklabels(
            labels=np.arange(ymin, ymax + 0.1, 0.1).round(1), fontsize=30
        )
    else:
        ax.set_ylabel("")
        ax.set_yticks([])
    ax.set_xlabel("")
    ax.get_legend().remove()
    plt.tight_layout()


def plot_conceptwise_performances(
    out_path: str,
    zeroshot_concept_errors: pd.DataFrame,
    probing_concept_errors: pd.DataFrame,
    dimensions: List[int],
    vice_embedding: Array,
    images: List[Any],
    verbose: bool = True,
) -> None:
    n_rows = 3
    f = plt.figure(figsize=(40, 34), dpi=150)
    gs = f.add_gridspec(n_rows, len(dimensions))
    for i in range(n_rows):
        for j, d in enumerate(dimensions):
            with sns.axes_style("ticks"):
                ax = f.add_subplot(gs[i, j])
                if i == 0:
                    dimension = vice_embedding[:, d]
                    visualize_dimension(
                        ax=ax,
                        images=images,
                        dimension=dimension,
                    )
                else:
                    if i == 1:
                        concept_subset = zeroshot_concept_errors[
                            zeroshot_concept_errors.dimension == d
                        ]
                        probing = False
                    else:
                        concept_subset = probing_concept_errors[
                            probing_concept_errors.dimension == d
                        ]
                        probing = True
                    plot_conceptwise_accuracies(
                        concept_subset=concept_subset,
                        ylabel=True if (j == 0) else False,
                        xlabel=True,
                        probing=probing,
                    )

    f.tight_layout()
    if not os.path.exists(out_path):
        print("\nOutput directory does not exist.")
        print("Creating output directory to save plot.\n")
        os.makedirs(out_path)

    plt.savefig(
        os.path.join(
            out_path,
            f"conceptwise_performance_{dimensions[0]:02d}_{dimensions[1]:02d}_{dimensions[2]:02d}.png",
        ),
        bbox_inches="tight",
    )
    if verbose:
        plt.show()
    plt.close()


def plot_probing_vs_zeroshot(results: pd.DataFrame, module: str, ylabel: bool) -> None:
    min = float(1 / 3)
    max = 0.6
    ax = sns.scatterplot(
        data=results,
        x="zero-shot",
        y="probing",
        hue="Training",  # marker color is determined by training objective
        style="Training",  # marker style is also determined by training objective
        s=400,
        alpha=0.9,
        legend="full",
        palette=PALETTE,
        markers=MARKERS,
    )
    ax.set_xlabel("Zero-shot odd-one-out accuracy", fontsize=30, labelpad=25)

    if ylabel:
        ax.set_ylabel("Probing odd-one-out accuracy", fontsize=30, labelpad=25)
    else:
        ax.set_ylabel("")

    ax.set_title(module.capitalize(), fontsize=32, pad=20)
    # make x and y limits equivalent
    ax.set_ylim([min, max])
    ax.set_xlim([min, max])
    # plot the x=y line
    ax.plot([min, max], [min, max], "--", alpha=0.8, color="grey", zorder=0)
    ax.set_xticks(np.arange(min, max, 0.02).round(2))
    ax.set_yticks(np.arange(min, max, 0.02).round(2))
    ax.set_xticklabels(np.arange(min, max, 0.02).round(2), fontsize=20)
    ax.set_yticklabels(np.arange(min, max, 0.02).round(2), fontsize=20)
    ax.legend(title="", ncol=1, loc="lower right", fancybox=True, fontsize=22)


def plot_probing_vs_zeroshot_performances(
    out_path: str,
    results: pd.DataFrame,
    modules: List[str],
    verbose: bool = True,
) -> None:
    """Plot probing against zero-shot odd-one-out accuracy for all models and both modules."""
    f = plt.figure(figsize=(28, 10), dpi=200)
    gs = f.add_gridspec(1, len(modules))
    sns.set_context("talk")
    with sns.axes_style("ticks"):
        for i, module in enumerate(modules):
            module_subset = results[results.module == module]
            ax = f.add_subplot(gs[0, i])
            plot_probing_vs_zeroshot(
                results=module_subset,
                module=module,
                ylabel=True if i == 0 else False,
            )
    f.tight_layout()

    if not os.path.exists(out_path):
        print("\nOutput directory does not exist.")
        print("Creating output directory to save plot.\n")
        os.makedirs(out_path)

    plt.savefig(
        os.path.join(
            out_path,
            f"probing_vs_zeroshot_performance.png",
        ),
        bbox_inches="tight",
    )
    if verbose:
        plt.show()
    plt.close()


def plot_logits_vs_penultimate(
    out_path: str,
    probing_results: pd.DataFrame,
    verbose: bool = True,
) -> None:
    min = 0.39
    max = 0.59
    plt.figure(figsize=(8, 6), dpi=100)
    sns.set_style("ticks")
    sns.set_context("paper")
    ax = sns.scatterplot(
        data=probing_results,
        x="probing_penultimate",
        y="probing_logits",
        hue="Architecture",  # marker color is determined by a model's base architecture
        style="Training",  # marker style is determined by training data/objective
        s=90,
        alpha=0.9,
        legend="full",
        palette=sns.color_palette(
            "colorblind", probing_results["Architecture"].unique().shape[0]
        ),
    )
    ax.set_xlabel("Penultimate", fontsize=18, labelpad=12)
    ax.set_ylabel("Logits", fontsize=18, labelpad=12)
    ax.set_ylim([min, max])
    ax.set_xlim([min, max])
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against each other
    ax.plot(lims, lims, "--", alpha=0.8, color="grey", zorder=0)
    ax.set_xticks(np.arange(min, max + 0.01, 0.02), fontsize=16)
    ax.set_yticks(np.arange(min, max + 0.01, 0.02), fontsize=16)
    ax.set_xticklabels(np.arange(min - 0.01, max, 0.02).round(2), fontsize=12)
    ax.set_yticklabels(np.arange(min - 0.01, max, 0.02).round(2), fontsize=12)
    ax.legend(title="", loc="upper left", ncol=2, fancybox=True, fontsize=11)
    ax.set_title("Probing odd-one-out accuracy", fontsize=18, pad=10)
    plt.tight_layout()

    if not os.path.exists(out_path):
        print("\nOutput directory does not exist.")
        print("Creating output directory to save plot.\n")
        os.makedirs(out_path)

    plt.savefig(
        os.path.join(
            out_path,
            f"penultimate_vs_logits.png",
        ),
        bbox_inches="tight",
    )
    if verbose:
        plt.show()
    plt.close()
