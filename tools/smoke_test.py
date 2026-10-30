"""Exercise the public processing stages using generated, disposable images."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image


def smoke(arguments):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from common.config import load_config, save_config
    from common.files import new_output
    scratch = new_output(arguments.scratch)
    settings = load_config(arguments.config)
    settings["imaging"].update(crop_xyxy=[0, 0, 64, 64], output_wh=[64, 64])
    settings["block_ksvd"]["dictionary"].update(atoms=8, iterations=1, sparsity_ratio=0.25)
    settings["training"].update(constant_epochs=1, decay_epochs=0, save_every=1, workers=0)
    config = scratch / "synthetic_parameters.json"
    save_config(settings, config)
    raw = scratch / "raw"
    y, x = np.mgrid[:64, :64]
    for folder in ("low", "high"):
        (raw / folder).mkdir(parents=True)
        for index in range(1, 4):
            radius = 8 if folder == "low" else 5
            pixels = 30 + x + 150 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / radius ** 2)
            Image.fromarray(np.clip(pixels + index, 0, 255).astype(np.uint8)).save(raw / folder / f"sample_{index:06d}.tif")
    def stage(name, *options):
        command = [sys.executable, "-B", str(root / "run.py"), "--config", str(config), name, *map(str, options)]
        subprocess.run(command, cwd=root, check=True)
    stage("prepare", "--input", raw, "--output", scratch / "prepared")
    stage("prior", "--input", scratch / "prepared/pgm/low", "--output", scratch / "priors", "--encoder", Path(arguments.encoder).resolve())
    stage("denoise", "--images", scratch / "prepared/images/low", "--priors", scratch / "priors",
        "--output", scratch / "denoised", "--profile", "cascade", "--matlab", arguments.matlab)
    stage("fixed", "--images", scratch / "prepared/images/low", "--output", scratch / "fixed",
        "--profile", "denoising", "--matlab", arguments.matlab)
    manifest = scratch / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_id", "split", "input", "reference"))
        for index, split in enumerate(("train", "val", "test"), 1):
            name = f"sample_{index:06d}"
            writer.writerow((name, split, f"images/low/{name}.tif", f"images/high/{name}.tif"))
    stage("train", "--manifest", manifest, "--data-root", scratch / "prepared", "--output", scratch / "model",
        "--variant", "full", "--device", arguments.device)
    stage("infer", "--input", scratch / "denoised/images", "--output", scratch / "enhanced",
        "--checkpoint", scratch / "model/usr_mamba_001.pt", "--device", arguments.device)
    annotations = scratch / "annotations"
    annotations.mkdir()
    np.savetxt(annotations / "sample_000003_rois.csv", [[12, 12, 32, 32]], delimiter=",", fmt="%d")
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 255
    Image.fromarray(mask).save(annotations / "sample_000003_mask.png")
    cases = scratch / "evaluation.csv"
    with cases.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "group", "reference", "result", "scale_map", "rois", "tu", "mask"))
        writer.writerow(("sample_000003", "point", "prepared/images/high/sample_000003.tif", "enhanced/sample_000003.tif",
            "denoised/scale_maps/sample_000003_continuous.csv", "annotations/sample_000003_rois.csv",
            "priors/sample_000003_tu.csv", "annotations/sample_000003_mask.png"))
    stage("evaluate", "--manifest", cases, "--data-root", scratch, "--output", scratch / "metrics",
        "--matlab", arguments.matlab, "--device", arguments.device)
    assert len(list((scratch / "fixed/images").rglob("*.tif"))) == 18
    assert len(list((scratch / "enhanced").glob("*.tif"))) == 3
    assert (scratch / "metrics/metrics.csv").is_file()
    summary = {"synthetic_only": True, "prepare": True, "prior": True, "adaptive": True,
        "fixed_sizes": 6, "train_epochs": 1, "inferred_images": 3, "evaluation": True}
    (scratch / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Synthetic processing chain passed. Results are outside the source archive.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "matlab", "encoder", "device", "scratch"):
        parser.add_argument("--" + name, required=True)
    smoke(parser.parse_args())
