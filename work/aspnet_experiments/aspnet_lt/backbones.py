import torch.nn as nn
from torchvision import models

from aspnet_lt.resnet_cifar import resnet32


class TorchvisionBackbone(nn.Module):
    def __init__(self, name='resnet18', feature_dim=64, pretrained=False):
        super().__init__()
        if name == 'resnet18':
            weights_cls = getattr(models, 'ResNet18_Weights', None)
            model_fn = models.resnet18
        elif name == 'resnet50':
            weights_cls = getattr(models, 'ResNet50_Weights', None)
            model_fn = models.resnet50
        else:
            raise ValueError(f'Unknown torchvision backbone: {name}')

        weights = weights_cls.DEFAULT if pretrained and weights_cls is not None else None
        try:
            net = model_fn(weights=weights)
        except TypeError:
            net = model_fn(pretrained=bool(pretrained))

        in_features = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        self.feature_dim = int(feature_dim)
        self.proj = nn.Identity() if in_features == self.feature_dim else nn.Linear(in_features, self.feature_dim)

    def forward(self, x):
        return self.proj(self.encoder(x))


def build_backbone(name='resnet32', feature_dim=64, pretrained=False):
    name = name.lower()
    if name == 'resnet32':
        if pretrained:
            raise ValueError('pretrained is not supported for resnet32')
        return resnet32(feature_dim=feature_dim)
    if name in {'resnet18', 'resnet50'}:
        return TorchvisionBackbone(name=name, feature_dim=feature_dim, pretrained=pretrained)
    raise ValueError(f'Unknown backbone: {name}')
