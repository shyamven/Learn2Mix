import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3 backbone with configurable output dimension."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNetImagenette(nn.Module):
    """ResNet18 used for Imagenette classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

