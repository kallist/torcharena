import torch
from torch import nn

from torcharena.config import RunConfig


class TinyCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(256, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class TinyResidualCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.stem = nn.Conv2d(1, 4, 3, padding=1)
        self.residual = nn.Sequential(nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
        self.head = nn.Linear(256, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        return self.head(torch.relu(hidden + self.residual(hidden)).flatten(1))


MODELS = {"tiny_cnn": TinyCNN, "tiny_resnet": TinyResidualCNN}


def make_model(config: RunConfig) -> nn.Module:
    return MODELS[config.model.name](config.model.num_classes)
