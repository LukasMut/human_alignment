import torch
from thingsvision.model_class import Model
import os
from torchvision import models
from torchvision import transforms


def load_vissl_r50(file, base_dir='vissl/models', grayscale=False, strict=True):
    state_dict = torch.load(os.path.join(base_dir, file), map_location=torch.device('cpu'))
    model = models.resnet50()
    if grayscale:
        model.conv1 = torch.nn.Conv2d(1, 64, 7, 1, 1, bias=False)
    model.fc = torch.nn.Identity()
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)
    return model


class CustomModel(Model):

    def __init__(self, ssl_models_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssl_models_path = ssl_models_path

    def load_model(self):
        """Load a pretrained *torchvision* or CLIP model into memory."""
        if self.backend == 'pt':
            if self.model_name == 'r50-barlowtwins':
                self.model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            elif self.model_name == 'r50-vicreg':
                self.model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
            elif self.model_name == 'r50-swav':
                self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            elif self.model_name == 'r50-simclr':
                self.model = load_vissl_r50(file='converted_vissl_simclr.torch', base_dir=self.ssl_models_path)
            elif self.model_name == 'r50-mocov2':
                self.model = load_vissl_r50(file='converted_vissl_mocov2.torch', base_dir=self.ssl_models_path)
            elif self.model_name == 'r50-jigsaw':
                self.model = load_vissl_r50(file='converted_vissl_jigsaw.torch', base_dir=self.ssl_models_path)
            elif self.model_name == 'r50-colorization':
                self.model = load_vissl_r50(file='converted_vissl_colorization.torch',
                                            grayscale=True, strict=False, base_dir=self.ssl_models_path)
            elif self.model_name == 'r50-rotnet':
                self.model = load_vissl_r50(file='converted_vissl_rotnet.torch', base_dir=self.ssl_models_path)
            else:
                super(CustomModel, self).load_model()
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            super(CustomModel, self).load_model()

    def get_transformations(self, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True):
        if self.model_name in ['r50-simclr', 'r50-mocov2', 'r50-jigsaw', 'r50-colorization', 'r50-rotnet']:
            # no normalization, Todo: check this carefully
            pipeline = transforms.Compose([
                transforms.Resize(resize_dim),
                transforms.CenterCrop(crop_dim),
                transforms.ToTensor(),
            ])
            return pipeline
        else:
            return super(CustomModel, self).get_transformations(resize_dim, crop_dim, apply_center_crop)
