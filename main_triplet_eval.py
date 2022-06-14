from dis import dis
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from models import load_model, get_normalization_for_model
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from functorch import vmap
from data.utils import load_dataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['things', 'cifar100', 'things-5k'])
parser.add_argument('--data_root')
parser.add_argument('--input-dim', default=224)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--device', default='cuda')
parser.add_argument('--out-file', default='results.csv')
args = parser.parse_args()

device = args.device


def entropy(p):
    return np.where(p > 0., p*np.log(p), 0.)

def vdot(X, Y):
    return vmap(lambda x, y: x @ y)(X, Y)

results = []
for model_name in tqdm(args.models):
    transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
    ])
    normalization = get_normalization_for_model(model_name)
    if normalization is not None:
        transform = transforms.Compose([transform, transforms.Normalize(*normalization)])

    dataset = load_dataset(name=args.dataset, data_dir=args.data_root, transform=transform)

    model = load_model(model_name)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    dl = DataLoader(dataset, batch_size=args.batch_size)
    correct, total = 0, 0
    triplet_probas = []
    with torch.no_grad():
        for x1, x2, x3, y in tqdm(dl):
            y = y.to(device)
            z = torch.stack([model(x.to(device)) for x in [x1, x2, x3]])
            distances = torch.zeros(3, x1.shape[0], device=device)
            similarities = torch.zeros_like(distances)
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                dist = 1 - F.cosine_similarity(z[i], z[j], dim=1)
                distances[i] += dist
                distances[j] += dist

                dots = vdot(z[i], z[j])
                similarities[i] += dots
                similarities[j] += dots

            batch_probas = F.softmax(similarities, dim=0)
            triplet_probas.extend(batch_probas.T.cpu().tolist())

            odd_one_out_idx = torch.argmax(distances, dim=0)
            correct += (odd_one_out_idx == y).sum()
            total += x1.shape[0]
    
    triplet_probas = np.asarray(triplet_probas)
    triplet_entropies = np.apply_along_axis(entropy, axis=1, arr=triplet_probas)
    accuracy = round((correct / total * 100).cpu().numpy().item(), 2)
    print(model_name, accuracy)
    results.append({
        'model': model_name,
        'accuracy': accuracy,
        'mean_entropy': np.mean(triplet_entropies),
        'median_entropy': np.median(triplet_entropies),
        'entropies': triplet_entropies,
    })
results = pd.DataFrame(results)
print(results)
results.to_csv(args.out_file)
