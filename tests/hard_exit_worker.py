"""Dedicated subprocess fixture: exit without exception/finally/signal cleanup."""

import os
import sys
from pathlib import Path

from torcharena.config import RunConfig
from torcharena.service import train
from torcharena.storage import Repository


def terminate(trainer, phase):
    if phase == "after_step" and trainer.progress["global_step"] == 3:
        os._exit(17)


train(Repository(Path(sys.argv[1])), RunConfig(), demo_hook=terminate)
