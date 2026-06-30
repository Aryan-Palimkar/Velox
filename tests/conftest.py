import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_collection_modifyitems(config, items):
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return torch.device("cuda")
