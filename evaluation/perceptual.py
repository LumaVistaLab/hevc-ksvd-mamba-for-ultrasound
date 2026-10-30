"""LPIPS on aligned grayscale images, using a configured feature network."""
import numpy as np
from PIL import Image
import torch


def perceptual_distance(result, reference, network, device):
    import lpips
    tensors = []
    for path in (result, reference):
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError("Perceptual evaluation expects 8-bit grayscale images")
            pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
        tensors.append(torch.from_numpy(pixels)[None, None].repeat(1, 3, 1, 1).to(device))
    if tensors[0].shape != tensors[1].shape:
        raise ValueError("Evaluation images must be aligned")
    model = lpips.LPIPS(net=network).to(device).eval()
    with torch.no_grad():
        return model(*tensors).item()
