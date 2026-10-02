import contextlib
import os
import platform
import random
import subprocess
import sys

import torch

from torcharena.config import RunConfig


def device_for(config: RunConfig) -> torch.device:
    requested = config.training.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose training.device: cpu")
    device = torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
    if config.training.amp and device.type != "cuda":
        raise ValueError("V0.1 AMP requires CUDA; set amp: false on CPU")
    return device


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.set_num_threads(1)


def capture_rng() -> dict:
    result = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": None,
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np
    except ImportError:
        return result
    result["numpy"] = np.random.get_state()
    return result


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["numpy"] is not None:
        import numpy as np

        np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextlib.contextmanager
def isolated_rng():
    state = capture_rng()
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    threads = torch.get_num_threads()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        restore_rng(state)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.set_num_threads(threads)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def environment(device: torch.device) -> dict:
    def git(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sha, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_sha": sha,
        "git_dirty": status != "" if status is not None else None,
    }
