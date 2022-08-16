import pickle
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pathlib
import os

COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'brown', 'yellow', 'pink']


def load_dataframe(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='resources/runs/things/penultimate/results.pkl')
    parser.add_argument('--aligned-path', default='resources/runs/things/penultimate/results.pkl')
    parser.add_argument('--network_data_path', default=None)
    parser.add_argument('--title', default='Logits')
    parser.add_argument('--output')
    args = parser.parse_args()

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    results = load_dataframe(args.path)
    aligned_results = load_dataframe(args.aligned_path)

    aligned_results['aligned_accuracy'] = aligned_results['accuracy']
    aligned_results = aligned_results[['aligned_accuracy', 'model']]

    fig, ax = plt.subplots()
    df = results.merge(networks, on='model').merge(aligned_results, on='model')
    df.loc[df.arch.isna(), 'arch'] = 'other'

    print(df)

    for i, group in enumerate(df['arch'].unique()):
        df[df.arch == group].plot(x='accuracy', y='aligned_accuracy', ax=ax,
                                  label=group, marker='x', kind='scatter',
                                  c=COLORS[i])

    plt.xlabel('Things')
    plt.ylabel('Things aligned')
    plt.grid()
    plt.savefig(args.output)
