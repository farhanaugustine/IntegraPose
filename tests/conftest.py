import re
import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path(request) -> Path:
    """Keep test temp dirs inside the repo to avoid global temp-root issues."""
    base_dir = Path(__file__).resolve().parent / "_tmp_pytest"
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    temp_dir = base_dir / f"{safe_name[:48]}_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
