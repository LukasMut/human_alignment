import os.path
from typing import List
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy', ]


def is_image_file(filename, mode='img'):
    if mode == 'img':
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif mode == 'np':
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


def make_dataset(dirs, mode='img'):
    if not isinstance(dirs, list):
        dirs = [dirs, ]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


class TwoAFCDataset(torch.utils.data.Dataset):
    def __init__(self, data_roots: List[str], transform=None):
        self.roots = data_roots
        self.transform = transform

        # image directory
        dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        ref_paths = make_dataset(dir_ref)
        ref_paths = sorted(ref_paths)

        dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        p0_paths = make_dataset(dir_p0)
        p0_paths = sorted(p0_paths)

        dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        p1_paths = make_dataset(dir_p1)
        p1_paths = sorted(p1_paths)

        # judgement directory
        dir_J = [os.path.join(root, 'judge') for root in self.roots]
        judge_paths = make_dataset(dir_J, mode='np')
        judge_paths = sorted(judge_paths)

        assert len(ref_paths) == len(p0_paths) and len(ref_paths) == len(p0_paths)
        self.paths = []
        self.triplets = []
        paths_idx = 0
        for idx, judge_path in tqdm(enumerate(judge_paths)):
            judge_path = judge_paths[idx]
            judge_img = np.load(judge_path).reshape((1, 1, 1,))  # [0,1]
            judge = torch.FloatTensor(judge_img).numpy().item()
            if judge == 0:
                # p0 is preferred
                correct_idx = paths_idx + 1
                odd_one_out_idx = paths_idx + 2
            elif judge == 1:
                correct_idx = paths_idx + 2
                odd_one_out_idx = paths_idx + 1
            else:
                # we skip this triplet because humans are unsure
                continue

            self.triplets.append((paths_idx, correct_idx, odd_one_out_idx))
            paths_idx += 3
            self.paths += [ref_paths[idx], p0_paths[idx], p1_paths[idx]]

        print(f'Total samples: {len(ref_paths)}, after filtering: {len(self.triplets)}')
        self.triplets = np.array(self.triplets)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.paths)

    def get_triplets(self):
        return self.triplets
