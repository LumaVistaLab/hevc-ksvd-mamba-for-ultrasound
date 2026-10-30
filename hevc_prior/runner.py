"""Extract TU and PU priors with configured HEVC search weights."""
from pathlib import Path
import subprocess
from common.files import image_files, new_output


def extract(settings, executable, input_root, output_root):
    root, files = image_files(input_root, (".pgm",))
    output = new_output(output_root, root)
    executable = Path(executable).resolve(strict=True)
    parameters = settings["hevc"]
    keys = ("quality_index", "gradient_threshold", "boost_offset", "boost_limit", "distortion_weight_limit",
        "rate_weight_floor", "zero_multiplier", "zero_shift", "zero_reduction_shift", "structure_weight")
    values = [parameters[key] for key in keys] + parameters["distortion_weights"] + parameters["rate_weights"]
    if len(values) != 20 or any(type(value) is not int for value in values):
        raise ValueError("HEVC expects twenty integral search parameters")
    temporary = output / "search_parameters.txt"
    temporary.write_text(" ".join(map(str, values)) + "\n", encoding="ascii")
    try:
        for source in files:
            target = (output / source.relative_to(root)).with_suffix(".hevc")
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(executable), str(source), str(target), str(temporary)], check=True)
            if not all(target.with_name(target.stem + suffix).is_file() for suffix in ("_tu.csv", "_pu.csv")):
                raise RuntimeError("Incomplete structural prior export")
    finally:
        temporary.unlink(missing_ok=True)
    return len(files)
