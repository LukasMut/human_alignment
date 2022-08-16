import pickle
import matplotlib.pyplot as plt
import argparse


def load_dataframe(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='resources/results/things/penultimate/results.pkl')
    parser.add_argument('--output')
    args = parser.parse_args()

    results = load_dataframe(args.path)
    fig, ax = plt.subplots()

    results.plot(x='layer', y='accuracy', legend=False)
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig(args.output)
