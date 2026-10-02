"""Built-in tensors and epoch-specific order use isolated torch generators."""

import torch

from torcharena.config import RunConfig


def synthetic(config: RunConfig, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in {"train", "val"}:
        raise ValueError("Unknown dataset split")
    n = config.dataset.train_samples if split == "train" else config.dataset.val_samples
    generator = torch.Generator().manual_seed(config.experiment.seed + (split == "val"))
    targets = torch.arange(n) % config.dataset.num_classes
    images = torch.randn(n, 1, 8, 8, generator=generator) * 0.08
    # A class-specific bright column makes the classification task deliberately learnable.
    images[torch.arange(n), 0, :, targets] += 1.0
    return images, targets.long()


DATASETS = {"synthetic_image_classification": synthetic}


def make_dataset(config: RunConfig, split: str):
    return DATASETS[config.dataset.name](config, split)


def epoch_order(config: RunConfig, epoch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.experiment.seed + 10_000 + epoch)
    return torch.randperm(config.dataset.train_samples, generator=generator)
