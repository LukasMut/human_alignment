import torch
from models import models_vit
import torch
from timm.models.layers import trunc_normal_
import timm


class ModelWrapper(torch.nn.Module):

    def __init__(self, model, features=768):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(features, 1)
        # trunc_normal_(self.fc, std=2e-5)

    def forward(self, x):
        return self.fc(self.model.forward_features(x))


def supervised_vit():
    return ModelWrapper(timm.create_model('vit_base_patch16_224', pretrained=True), features=768)


def random_vit():
    return ModelWrapper(timm.create_model('vit_base_patch16_224', pretrained=False), features=768)


def load_vit_model(chkpt_dir, arch='vit_base_patch16', state_dict_key=None):
    model = getattr(models_vit, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    state_dict = checkpoint[state_dict_key] if state_dict_key is not None else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    return ModelWrapper(model=model)
