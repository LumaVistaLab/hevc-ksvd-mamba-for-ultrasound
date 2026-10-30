"""Paired beamline images with explicit source separation and paired augmentation."""
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image
import torch
from common.files import contained_file


@dataclass(frozen=True)
class ImagePair:
    identifier: str
    split: str
    low: Path
    high: Path


def read_manifest(path, data_root):
    pairs, identifiers, file_splits = [], set(), {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["source_id", "split", "input", "reference"]:
            raise ValueError("Expected source_id,split,input,reference columns")
        for row in reader:
            identifier, split = row["source_id"], row["split"]
            if not re.fullmatch(r"sample_[0-9]{6}", identifier) or split not in ("train", "val", "test"):
                raise ValueError("Invalid neutral source identifier or split")
            if identifier in identifiers:
                raise ValueError("Each source must appear in exactly one dataset row")
            identifiers.add(identifier)
            low = contained_file(data_root, row["input"])
            high = contained_file(data_root, row["reference"])
            if low == high or low.suffix.lower() not in (".tif", ".tiff") or high.suffix.lower() not in (".tif", ".tiff"):
                raise ValueError("Supply distinct paired TIFF files")
            for image in (low, high):
                if image in file_splits and file_splits[image] != split:
                    raise ValueError("An image cannot belong to multiple dataset splits")
                file_splits[image] = split
            pairs.append(ImagePair(identifier, split, low, high))
    if not pairs:
        raise ValueError("The dataset manifest is empty")
    return pairs


class PairedImages(torch.utils.data.Dataset):
    def __init__(self, pairs, split, augmentation, seed, divisor):
        self.samples = []
        self.flip = split == "train" and augmentation["horizontal_flip"]
        rng = random.Random(seed)
        filters = {"lanczos": Image.Resampling.LANCZOS, "bilinear": Image.Resampling.BILINEAR}
        self.resampling = filters[augmentation["crop_resampling"]]
        for pair in pairs:
            if pair.split != split:
                continue
            with Image.open(pair.low) as low, Image.open(pair.high) as high:
                width, height = low.size
                if low.mode != "L" or high.mode != "L" or low.size != high.size:
                    raise ValueError("Paired images must be aligned 8-bit grayscale")
                if min(width, height) < divisor * 2 or width % divisor or height % divisor:
                    raise ValueError("Invalid dimensions for the configured encoder")
            count = augmentation["crop_count"] if split == "train" else 1
            if count < 1:
                raise ValueError("Crop count must be positive")
            for _ in range(count):
                crop = (0, 0, 0)
                limit = augmentation["crop_max_pixels"] if split == "train" else 0
                if limit < 0:
                    raise ValueError("Crop limit cannot be negative")
                if limit:
                    while not any(crop):
                        left = rng.randint(0, min(limit, width - 2))
                        crop = (left, rng.randint(0, min(limit, width - left - 1)), rng.randint(0, min(limit, height - 1)))
                self.samples.append((pair, crop))

    def __len__(self):
        return len(self.samples) * (2 if self.flip else 1)

    def __getitem__(self, index):
        pair, (left, right, bottom) = self.samples[index % len(self.samples)]
        images = []
        for path in (pair.low, pair.high):
            with Image.open(path) as image:
                if any((left, right, bottom)):
                    size = image.size
                    image = image.crop((left, 0, image.width - right, image.height - bottom)).resize(size, self.resampling)
                pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
            tensor = torch.from_numpy(pixels).unsqueeze(0)
            images.append(tensor.flip(2) if index >= len(self.samples) else tensor)
        return images[0], images[1], pair.identifier
