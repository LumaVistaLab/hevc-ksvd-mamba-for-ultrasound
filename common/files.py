"""Neutral identifiers and explicit, contained dataset paths."""
from pathlib import Path, PurePosixPath
import re


def neutral_path(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Expected a relative path")
    if not re.fullmatch(r"[a-z0-9_/]+\.[a-z0-9]+", relative):
        raise ValueError("Use neutral lowercase paths")
    return path


def contained_file(root, relative):
    relative = neutral_path(relative)
    root = Path(root).resolve(strict=True)
    target = root.joinpath(*relative.parts).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Input must be a file inside the data root")
    return target


def image_files(root, suffixes=(".tif", ".tiff")):
    root = Path(root).resolve(strict=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        raise ValueError("No input images")
    for path in files:
        neutral_path(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise ValueError("Dataset links are not accepted")
    return root, files


def new_output(path, input_root=None):
    target = Path(path).resolve()
    if input_root is not None and target.is_relative_to(Path(input_root).resolve()):
        raise ValueError("Output must be outside the input tree")
    target.mkdir(parents=True, exist_ok=False)
    return target
