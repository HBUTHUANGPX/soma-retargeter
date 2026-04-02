import os
from pathlib import Path


def as_newton_usd_source(source: str | os.PathLike[str]) -> str:
    """Return a string path so newton.add_usd() takes the file-open code path."""
    return str(Path(source))
