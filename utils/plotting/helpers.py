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
    "Supervised (ImageNet-1K)": "coral",
    "Supervised (Ecoset)": "brown",
    "Supervised (ImageNet-21K)": "darkcyan",
    "Supervised (JFT-3B)": "black",
    "Self-Supervised": "darkgreen",
    # "VICE": "red",
}

MARKERS = {
    "Image/Text": "o",
    "Supervised (ImageNet-1K)": "s",
    "Supervised (Ecoset)": "^",
    "Supervised (ImageNet-21K)": "P",
    "Supervised (JFT-3B)": "X",
    "Self-Supervised": "D",
    # "VICE": "*",
}

CONCEPT_MAPPING = {
        0: 'Metal',
        1: 'Food',
        2: 'Plant-related', 
        3: 'Animal-related',
        4: 'Furniture',
        5: 'Clothing',
        6: 'Royal',
        7: 'Outdoors-related',
        8: 'Body part',
        9: 'Vehicles',
        10: 'Wood',
        11: 'Tools',
        12: 'Technology',
        13: 'Colorful',
        14: 'Patterns',
        15: 'Circular',
        16: 'Sports',
        17: 'Paper',
        18: 'Liquids',
        19: 'Sea',
        20: 'Red',
        21: 'Powdery',
        22: 'Hygiene',
        23: 'Weapons',
        24: 'Has-grating',  # switch w 33?
        25: 'Black',
        26: 'Sky-related',
        27: 'Long/thin',
        28: 'White',
        29: 'Decorative',  # feminine (stereotypically) fits better but is too long
        30: 'Spherical',
        31: 'Green',  # Looks like green, but there is no such dimension?
        32: 'Musical instrument',  # dropped the "related"
        33: 'Patterned',  # switch w 24?
        34: 'Bugs',
        35: 'Fire-related',
        36: 'Shiny',
        37: 'String-related',
        38: 'Arms/legs/skin',  # dropped the related
        39: 'Elongated',  # confounded w "long"
        40: 'Home-related',  # really no idea
        41: 'Toy-related',
        42: 'Yellow',  # this seems to be mixed w yellow
        43: 'Medicine-related',
        44: 'Ice/Winter',
    }
    
def set_context() -> None:
    sns.set_context("paper")


def concat_images(images: Array, top_k: int) -> Array:
    img_combination = np.concatenate([
        np.concatenate([img for img in images[:int(top_k/2)]], axis = 1),
        np.concatenate([img for img in images[int(top_k/2):]], axis = 1)], axis = 0)
    return img_combination

def visualize_dimension(
    ax: Any, 
    images: Array, 
    dimension: Array,
    d: int,
    top_k: int = 6
) -> None:
    # sort dimension by weights in decending order and get top-k objects
    topk_objects = np.argsort(-dimension)[:top_k]
    topk_images = images[topk_objects]
    img_comb = concat_images(images=topk_images, top_k=top_k)
    set_context()
    for spine in ax.spines:
        ax.spines[spine].set_color(rgb.tue_dark)
        ax.spines[spine].set_linewidth(8)
    ax.imshow(img_comb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'Dimension {d+1} / {CONCEPT_MAPPING[d]}', fontsize=62, pad=30, color='dimgrey')
    
def plot_conceptwise_accuracies(
    concept_subset: pd.DataFrame, 
    ylabel: bool, 
    xlabel: bool,
    probing: bool,
) -> None:
    # sort models by their odd-one-out accuracy in descending order
    ymin = .2
    ymax = .9
    concept_subset.sort_values(by=['odd-one-out-accuracy'], axis=0, ascending=False, inplace=True)
    set_context()
    ax = sns.swarmplot(
        data=concept_subset,
        x="family", 
        y="odd-one-out-accuracy",
        hue="training",
        orient="v",
        edgecolor="gray",
        s=16,
        alpha=.6,
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
            "Probing odd-one-out accuracy"
            if probing
            else "Zero-shot odd-one-out accuracy"
        )
        ax.set_ylabel(label, fontsize=40, labelpad=30)
        ax.set_yticklabels(
            labels=np.arange(ymin, ymax + 0.1, 0.1).round(1), fontsize=38
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
    verbose: bool = True 
) -> None:
    nrow = 3
    ncol = len(dimensions)
    f = plt.figure(figsize=(40, 32), dpi=150)
    gs = f.add_gridspec(nrow, ncol)
    for i in range(nrow):
        for j, d in enumerate(dimensions):
            with sns.axes_style("white"):
                ax = f.add_subplot(gs[i, j])
                if i == 0:
                    dimension = vice_embedding[:, d]
                    visualize_dimension(
                        ax=ax,
                        images=images,
                        dimension=dimension,
                        d=d,
                    )
                else:
                    if i == 1:
                        concept_subset = zeroshot_concept_errors[zeroshot_concept_errors.dimension==d]
                        probing = False
                    else:
                        concept_subset = probing_concept_errors[probing_concept_errors.dimension==d]
                        probing = True
                    plot_conceptwise_accuracies(
                        concept_subset=concept_subset, 
                        ylabel=True if j == 0 else False,
                        xlabel=True,
                        probing=probing,
                    )
    f.tight_layout()
    if not os.path.exists(out_path):
        print('\nOutput directory does not exist.')
        print('Creating output directory to save plot.\n')
        os.makedirs(out_path)

    plt.savefig(os.path.join(out_path, f'conceptwise_performance_{dimensions[0]:02d}_{dimensions[1]:02d}_{dimensions[2]:02d}.png'), bbox_inches='tight')
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
        alpha=0.6,
        legend="full",
        palette=PALETTE,
        markers=MARKERS,
    )
    ax.set_xlabel("Zero-shot odd-one-out accuracy", fontsize=35, labelpad=25)

    if ylabel:
        ax.set_ylabel("Probing odd-one-out accuracy", fontsize=35, labelpad=25)
    else:
        ax.set_ylabel("")

    ax.set_title(module.capitalize(), fontsize=40, pad=20)
    # set x and y limits to be the same
    ax.set_ylim([min, max])
    ax.set_xlim([min, max])
    # plot the x=y line
    ax.plot([min, max], [min, max], "--", alpha=0.8, color="grey", zorder=0)
    ax.set_xticks(np.arange(min, max, 0.04).round(2))
    ax.set_yticks(np.arange(min, max, 0.04).round(2))
    ax.set_xticklabels(np.arange(min, max, 0.04).round(2), fontsize=22)
    ax.set_yticklabels(np.arange(min, max, 0.04).round(2), fontsize=22)
    ax.legend(title="", ncol=1, loc="lower right", fancybox=True, fontsize=24)


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
        alpha=0.7,
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



def alignment_plot(
    agreements: pd.DataFrame,
    layer: str,
    comparison: str,
    title: bool,
    ylabel: bool,
    cbar: bool,
) -> None:
    min_agreement = agreements.to_numpy().min()
    sns.set_context("paper")
    # here set the scale by 3
    sns.set(font_scale=1.3)
    sns.heatmap(
                data=agreements,
                vmin=0, #min_agreement, # TODO: set vmin to 0 or minimum agreement percentage?
                vmax=float(1),
                annot=False,
                cbar=False, #True if cbar else False,
                square=True,
                xticklabels=True, #'auto',
                yticklabels=True, #'auto',
                alpha=.8, #.9
                cmap=sns.color_palette("flare", as_cmap=True),
    )
    plt.xlabel('')
    
    if ylabel:
        plt.ylabel(layer.capitalize(), fontsize=35, labelpad=30)
    else:
        plt.ylabel('')
        
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    if title:
        if comparison.lower() == 'cka':
            title = 'Centered Kernel Alignment (CKA)' 
        else:
            title = 'Odd-one-out choice agreement'
        plt.title(title, fontsize=30, pad=20)