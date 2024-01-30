from pathlib import Path

import pytest


@pytest.fixture
def data_folder(request) -> Path:
    folder_path = Path(__file__).parent / 'data'
    assert folder_path.exists()
    request.cls.data_folder = folder_path
    return folder_path
