
import torch
import os
from torchvision import models

def load_vissl_r50(file, base_dir='vissl/models', grayscale=False, strict=True):
    state_dict = torch.load(os.path.join(base_dir, file), map_location=torch.device('cpu'))
    model = models.resnet50()
    if grayscale:
        model.conv1 = torch.nn.Conv2d(1, 64, 7, 1, 1, bias=False)
    model.fc = torch.nn.Identity()
    msg = model.load_state_dict(state_dict, strict=strict)
    print(f'\n{msg}\n')
    return model