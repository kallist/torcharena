"""Closed, validated configuration; every field is resume-critical in V0.1."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Experiment(StrictModel):
    name: str = Field(default="tiny-cnn-demo", min_length=1, max_length=120)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class Dataset(StrictModel):
    name: Literal["synthetic_image_classification"] = "synthetic_image_classification"
    train_samples: int = Field(default=64, ge=1, le=100_000)
    val_samples: int = Field(default=32, ge=1, le=100_000)
    num_classes: int = Field(default=2, ge=2, le=8)


class Model(StrictModel):
    name: Literal["tiny_cnn", "tiny_resnet"] = "tiny_cnn"
    num_classes: int = Field(default=2, ge=2, le=8)


class Training(StrictModel):
    epochs: int = Field(default=4, ge=1, le=10_000)
    batch_size: int = Field(default=16, ge=1, le=4096)
    learning_rate: float = Field(default=0.02, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    amp: bool = False
    deterministic: bool = True

    @model_validator(mode="after")
    def finite_lr(self):
        if not math.isfinite(self.learning_rate):
            raise ValueError("training.learning_rate must be finite and positive")
        return self


class Checkpoint(StrictModel):
    every_n_steps: int = Field(default=2, ge=1)
    keep_last: int = Field(default=2, ge=1, le=20)
    save_best: bool = True


class EarlyStopping(StrictModel):
    enabled: bool = False
    metric: Literal["val_loss", "val_accuracy"] = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=3, ge=1)


class RunConfig(StrictModel):
    experiment: Experiment = Field(default_factory=Experiment)
    dataset: Dataset = Field(default_factory=Dataset)
    model: Model = Field(default_factory=Model)
    training: Training = Field(default_factory=Training)
    checkpoint: Checkpoint = Field(default_factory=Checkpoint)
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)

    @model_validator(mode="after")
    def compatible_classes(self):
        if self.model.num_classes != self.dataset.num_classes:
            raise ValueError(
                f"model.num_classes = {self.model.num_classes}, "
                f"dataset.num_classes = {self.dataset.num_classes}; classes must match"
            )
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class UniqueSafeLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate YAML mapping keys."""


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_config(path: Path) -> RunConfig:
    if path.stat().st_size > 1_000_000:
        raise ValueError("Config exceeds 1 MB")
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueSafeLoader)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return RunConfig.model_validate(data)
