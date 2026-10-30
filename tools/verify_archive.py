"""Verify source scope, neutral paths, imports and the SHA-256 inventory."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


ROOT_FILES = {".gitignore", "README.md", "README_zh-CN.md", "LICENSE", "CMakeLists.txt", "run.py",
              "requirements.txt", "requirements-perceptual.txt", "thesis.pdf"}
DIRECTORIES = {"configs": {".json"}, "common": {".py"}, "imaging": {".py", ".m"},
    "hevc_prior": {".c", ".h", ".py"}, "block_ksvd": {".m"}, "usr_mamba": {".py"},
    "evaluation": {".py", ".m"}, "licenses": {".txt"}, "docs": {".md", ".json"}, "tools": {".py"}}


def verify(root, update):
    root = Path(root).resolve()
    files = [root / name for name in ROOT_FILES]
    for directory, extensions in DIRECTORIES.items():
        for path in (root / directory).rglob("*"):
            if not path.is_file():
                continue
            if path.parent != root / directory or path.suffix not in extensions or not re.fullmatch(r"[a-z0-9_]+\.[a-z]+", path.name):
                raise ValueError("Unexpected source entry: " + path.relative_to(root).as_posix())
            files.append(path)
    records = []
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Archive links are not accepted")
        data = path.read_bytes()
        if relative == "thesis.pdf":
            if not data.startswith(b"%PDF-"):
                raise ValueError("Invalid thesis PDF")
        else:
            text = data.decode("utf-8-sig")
            if "\0" in text:
                raise ValueError("Unexpected binary payload")
            patterns = (r"(?i)(?<![a-z0-9_])[a-z]:[\\/]", r"(?i)/(?:users|home|mnt)/",
                        r"AKIA[A-Z0-9]{16}", r"gh[pousr]_[a-zA-Z0-9]{20,}")
            if any(re.search(pattern, text) for pattern in patterns):
                raise ValueError("Potential private path or credential: " + relative)
            if path.suffix == ".py":
                tree = ast.parse(text, filename=relative)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level and node.module:
                        module = path.parent.joinpath(*node.module.split("."))
                        if not module.with_suffix(".py").is_file():
                            raise ValueError("Unresolved local import: " + relative)
            if path.suffix == ".m":
                first = text.splitlines()[0]
                if not re.search(r"\b" + path.stem + r"\s*\(", first):
                    raise ValueError("MATLAB entry name does not match its file")
        if relative != "docs/archive_manifest.json":
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"version": 1, "binary_exception": "thesis.pdf", "files": records}
    target = root / "docs/archive_manifest.json"
    if update:
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif json.loads(target.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Inventory differs; review changes before updating")
    print(f"Verified {len(records)} entries; the thesis is the only permitted binary.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    arguments = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], arguments.update)
