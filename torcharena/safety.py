import torch

from torcharena.config import RunConfig
from torcharena.data import make_dataset
from torcharena.models import make_model
from torcharena.runtime import device_for, isolated_rng, seed_all


class TrainingError(RuntimeError):
    def __init__(self, condition: str, message: str):
        self.condition = condition
        super().__init__(message)


class TrainingGuard:
    @staticmethod
    def finite(tensor: torch.Tensor, condition: str) -> None:
        if not torch.isfinite(tensor).all():
            raise TrainingError(condition, f"Detected {condition}; inspect data and learning rate")

    @classmethod
    def gradients(cls, model):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients:
            raise TrainingError("missing_gradients", "No trainable parameter received a gradient")
        for gradient in gradients:
            cls.finite(gradient, "non_finite_gradient")

    @classmethod
    def parameters(cls, model):
        for parameter in model.parameters():
            cls.finite(parameter, "non_finite_parameter")


def preflight(config: RunConfig) -> dict:
    with isolated_rng():
        seed_all(config.experiment.seed, config.training.deterministic)
        device = device_for(config)
        model = make_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        for split in ("train", "val"):
            images, targets = make_dataset(config, split)
            if len(images) == 0:
                raise TrainingError("empty_dataset", f"{split} dataset is empty")
            images, targets = (
                images[: config.training.batch_size],
                targets[: config.training.batch_size],
            )
            if images.ndim != 4 or targets.ndim != 1 or len(images) != len(targets):
                raise TrainingError("invalid_batch", "Expected images [N,C,H,W] and targets [N]")
            with torch.autocast(device.type, enabled=config.training.amp):
                output = model(images.to(device))
                if output.shape != (len(images), config.dataset.num_classes):
                    raise TrainingError("output_shape", "Model output must be [N,num_classes]")
                loss = torch.nn.functional.cross_entropy(output, targets.to(device))
            TrainingGuard.finite(loss, "non_finite_loss")
            optimizer.zero_grad(set_to_none=True)
            scaler = torch.amp.GradScaler(device.type, enabled=config.training.amp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            TrainingGuard.gradients(model)
            scaler.step(optimizer)
            scaler.update()
            TrainingGuard.parameters(model)
    return {
        key: "PASS"
        for key in ["Config", "Dataset", "Batch", "Forward", "Loss", "Backward", "Optimizer step"]
    } | {"Device": str(device), "AMP": "ENABLED" if config.training.amp else "DISABLED"}
