"""Apply an explicitly selected USR-Mamba checkpoint to a neutral image tree."""
import numpy as np
from PIL import Image
import torch
from common.files import image_files, new_output
from .network import USRMamba


def enhance(checkpoint, input_root, output_root, device):
    root, files = image_files(input_root)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = USRMamba(state["model_config"], state["variant"]).to(device)
    model.load_state_dict(state["generator"], strict=True)
    model.eval()
    output = new_output(output_root, root)
    with torch.no_grad():
        for path in files:
            with Image.open(path) as image:
                if image.mode != "L":
                    raise ValueError("Inference requires 8-bit grayscale inputs")
                pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
            prediction = model(torch.from_numpy(pixels)[None, None].to(device))[0, 0].cpu().numpy()
            pixels = np.clip((prediction + 1) * 127.5, 0, 255).astype(np.uint8)
            target = (output / path.relative_to(root)).with_suffix(".tif")
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(target)
    return len(files)
