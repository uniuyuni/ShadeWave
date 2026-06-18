from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external"


def external_path(*parts: str) -> Path:
    return EXTERNAL_ROOT.joinpath(*parts)


def add_external_path(*parts: str) -> Path:
    path = external_path(*parts).resolve()
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path
