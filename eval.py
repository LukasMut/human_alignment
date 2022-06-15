import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from models import load_model, get_normalization_for_model
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from data.utils import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['things', 'cifar100-fine', 'things-5k',
                                                                        'cifar100-coarse', 'cifar10', 'things-new'])
parser.add_argument('--data_root')
parser.add_argument('--input-dim', default=224)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--device', default='cuda')
parser.add_argument('--out-file', default='results.csv')
parser.add_argument('--imagenet-logits', action='store_true')
args = parser.parse_args()

device = args.device

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
    if not args.imagenet_logits:
        model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for x in tqdm(dl):
            x = x.to(device)
            repr = model(x)
            embeddings.append(repr.detach().cpu())
    embeddings = torch.cat(embeddings, dim=0)

    correct, total = 0, 0
    print('embeddings', embeddings.shape)
    for i1, i2, i3 in dataset.get_triplets():
        z = torch.stack([embeddings[i] for i in [i1, i2, i3]])
        distances = torch.zeros(3, device=device)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            dist = 1 - F.cosine_similarity(z[i], z[j], dim=0)
            distances[i] += dist
            distances[j] += dist
        odd_one_out_idx = torch.argmax(distances, dim=0)
        correct += (odd_one_out_idx == 2).sum()
        total += 1

    accuracy = round((correct / total * 100).cpu().numpy().item(), 2)
    print(model_name, accuracy)
    results.append({
        'model': model_name,
        'accuracy': accuracy
    })
results = pd.DataFrame(results)
print(results)
results.to_csv(args.out_file)
