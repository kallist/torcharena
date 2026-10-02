import pytest

from torcharena.config import RunConfig
from torcharena.storage import Repository


@pytest.fixture
def config():
    return RunConfig()


@pytest.fixture
def repository(tmp_path):
    return Repository(tmp_path / "home")
