"""Create aligned 8-bit TIFF and PGM images from neutral local inputs."""
from PIL import Image
from common.files import image_files, new_output


def prepare(input_root, output_root, settings):
    root, files = image_files(input_root)
    width, height = settings["output_wh"]
    box = settings["crop_xyxy"]
    if len(box) != 4 or min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("HEVC dimensions must be positive multiples of 32")
    filters = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
               "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
    resampling = filters[settings["resampling"]]
    # Validate the batch before creating any results.
    for path in files:
        with Image.open(path) as image:
            if image.mode != "L" or image.n_frames != 1:
                raise ValueError("Expected single-frame 8-bit grayscale TIFF")
            left, top, right, bottom = box
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("Crop lies outside the image")
            if int(width * (bottom - top) / (right - left)) != height:
                raise ValueError("Configured output does not preserve the crop aspect ratio")
    destination = new_output(output_root, root)
    for path in files:
        relative = path.relative_to(root).with_suffix("")
        with Image.open(path) as image:
            resized = image.crop(box).resize((width, height), resampling)
            pixels = Image.frombytes("L", resized.size, resized.tobytes())
        for folder, extension in (("images", ".tif"), ("pgm", ".pgm")):
            target = (destination / folder / relative).with_suffix(extension)
            target.parent.mkdir(parents=True, exist_ok=True)
            pixels.save(target)
    return len(files)
