import torch
from thingsvision.model_class import Model
from torchvision import transforms
from .utils import load_vissl_r50

class CustomModel(Model):

    def __init__(self, ssl_models_path, *args, **kwargs):
        self.ssl_models_path = ssl_models_path
        super().__init__(*args, **kwargs)

    def load_model(self):
        """Load pretrained SSL models into memory."""
        if self.source in ['torchvision', 'timm']:
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
            # NOTE: NO normalization?
            # TODO: check this carefully!
            pipeline = transforms.Compose([
                transforms.Resize(resize_dim),
                transforms.CenterCrop(crop_dim),
                transforms.ToTensor(),
            ])
            return pipeline
        else:
            return super(CustomModel, self).get_transformations(resize_dim, crop_dim, apply_center_crop)