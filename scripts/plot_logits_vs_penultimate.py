import pickle
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pathlib
import os
from matplotlib.ticker import StrMethodFormatter

COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'brown', 'yellow', 'pink']


def load_dataframe(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='resources/all_results.pkl')
    parser.add_argument('--output')
    parser.add_argument('--mode', choices=['zero-shot', 'probing'], default='zero-shot')
    args = parser.parse_args()

    results = load_dataframe(args.results)
    print(len(results))
    logits = results[results.module == 'logits']
    penultimate = results[results.module == 'penultimate']

    df = logits.merge(penultimate, on='model')
    df.to_csv('test.csv', index=False)
    if args.mode == 'zero-shot':
        df['difference'] = round((df['zero-shot_x'] - df['zero-shot_y']) * 100, 2)
    else:
        df['difference'] = round((df['probing_x'] - df['probing_y']) * 100, 2)

    df = df[df.apply(lambda x: not (x['model'].startswith('r50') or x['model'].startswith('clip')), axis=1)]

    df = df.sort_values(by='difference')
    ax = df.plot.barh(x='model', y='difference', legend=False, figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)

    for loc in ['right', 'top', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)

    model_names = df['model'].values
    max_length = max(map(len, model_names))
    print(max_length)
    for i, p in enumerate(ax.patches):
        positive = p.get_width() > 0
        if not positive:
            p.set_color('#ff5252')
        ax.annotate("+%.2f" % p.get_width() if positive else "%.2f" % p.get_width(),
                    (p.get_x() + p.get_width(),
                     p.get_y()), xytext=(5 if positive else -30, 5),
                    color='#86bf91' if positive else '#ff5252',
                    alpha=1.0,
                    textcoords='offset points')

        if positive:
            x = p.get_x()
        else:
            x = p.get_x()
        """if positive:
            name = model_names[i].rjust(max_length, ' ')
            print(len(name))
        else:
            name = model_names[i]
        ax.annotate(name, (x,
                           p.get_y()), xytext=(-300 if positive else 10, 10),
                    textcoords='offset points')"""

    ax.tick_params(
        bottom=False, top=False,
        left=False, right=False)

    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    ax.set_xlabel("Accuracy Difference", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    # ax.set_ylabel("Model", labelpad=20, weight='bold', size=12)
    # ax.get_yaxis().set_visible(False)

    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=1.0, color='#eeeeee', zorder=0)
    plt.title("Penultimate/Logits")
    # plt.savefig(args.output)
    plt.tight_layout()
    # plt.show()
    if args.mode == 'probing':
        plt.savefig('resources/probing.png')
    else:
        plt.savefig('resources/zero-shot.png')
