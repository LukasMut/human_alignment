import torch.nn
from data.cifar100 import CIFAR100Triplet
from data.things import THINGSTriplet
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from models import load_model
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['things', 'cifar100'])
parser.add_argument('--data_root')
parser.add_argument('--input-dim', default=224)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

device = args.device

transform = transforms.Compose([
    transforms.Resize((args.input_dim, args.input_dim)),
    transforms.ToTensor(),
    transforms.Normalize(*IMAGENET_NORM)
])

if args.dataset == 'cifar100':
    # for all the imagenet models, we just use the train split for more samples
    dataset = CIFAR100Triplet(root='resources/datasets', train=True,
                            download=True, transform=transform,
                            samples=10000, seed=args.seed)
elif args.dataset == 'things':
    # for all the imagenet models, we just use the train split for more samples
    dataset = THINGSTriplet(root=args.data_root, train=True,
                            download=True, transform=transform)

for model_name in tqdm(args.models):
    model = load_model(model_name)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    dl = DataLoader(dataset, batch_size=args.batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for x1, x2, x3, y in tqdm(dl):
            y = y.to(device)
            z = torch.stack([model(x.to(device)) for x in [x1, x2, x3]])
            similarities = torch.zeros(3, x1.shape[0], device=device)
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                sim = F.cosine_similarity(z[i], z[j], dim=1)
                similarities[i] += sim
                similarities[j] += sim
            odd_one_out_idx = torch.argmin(similarities, dim=0)
            correct += (odd_one_out_idx == y).sum()
            total += x1.shape[0]
    print(model_name, round((correct / total * 100).cpu().numpy().item(), 2))
