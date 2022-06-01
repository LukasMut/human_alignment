import torchvision.models as models
import torch
import os
from .vit import load_vit_model, supervised_vit, random_vit
import timm

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class ModelWrapper(torch.nn.Module):

    def __init__(self, model, repre_dim=2048):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(repre_dim, 1)

    def forward(self, x):
        return self.fc(self.model(x))


def load_vissl_r50(file, base_dir='vissl/models', grayscale=False, strict=True):
    state_dict = torch.load(os.path.join(base_dir, file), map_location=torch.device('cpu'))
    model = models.resnet50()
    if grayscale:
        model.conv1 = torch.nn.Conv2d(1, 64, 7, 1, 1, bias=False)
    model.fc = torch.nn.Identity()
    model.load_state_dict(state_dict, strict=strict)
    return model


def load_model(name):
    if name == 'r18-random':
        return models.resnet18()
    elif name == 'r18-imagenet':
        return models.resnet18(pretrained=True)
    elif name == 'r50-barlow-twins':
        return torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    elif name == 'r50-random':
        return models.resnet50()
    elif name == 'r50-imagenet':
        return models.resnet50(pretrained=True)
    elif name == 'r50-dino':
        return torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif name == 'r50-swav':
        return torch.hub.load('facebookresearch/swav:main', 'resnet50')
    elif name == 'r50-vicreg':
        return ModelWrapper(torch.hub.load('facebookresearch/vicreg:main', 'resnet50'), repre_dim=2048)
    elif name == 'r50-jigsaw':
        return load_vissl_r50(file='converted_vissl_jigsaw.torch')
    elif name == 'r50-colorization':
        return load_vissl_r50(file='converted_vissl_colorization.torch', grayscale=True, strict=False)
    elif name == 'r50-rotnet':
        return load_vissl_r50(file='converted_vissl_rotnet.torch')
    elif name == 'r50-simclr':
        return load_vissl_r50(file='converted_vissl_simclr.torch')
    elif name == 'r50-mocov2':
        return load_vissl_r50(file='converted_vissl_mocov2.torch')
    elif name == 'r50-pirl':
        return load_vissl_r50(file='converted_vissl_pirl.torch')
    elif name == 'vit-b16-mae':
        chkpt_dir = 'mae_pretrain_vit_base.pth'
        return load_vit_model(chkpt_dir, 'vit_base_patch16', state_dict_key='model')
    elif name == 'vit-b16-dino':
        chkpt_dir = 'dino_vitbase16_pretrain.pth'
        return load_vit_model(chkpt_dir, 'vit_base_patch16')
    elif name == 'vit-b16-mocov3':
        chkpt_dir = 'mocov3-vit-b-300ep.pth'
        return load_vit_model(chkpt_dir, 'vit_base_patch16')
    elif name == 'vit-b16-supervised':
        return supervised_vit()
    elif name == 'vit-b16-random':
        return random_vit()
    else:
        raise ValueError('unknown model')


def get_normalization_for_model(name):
    if name == 'r50-colorization':
        return None
    return IMAGENET_NORM
