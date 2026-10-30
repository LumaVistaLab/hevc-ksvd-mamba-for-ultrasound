"""Invoke a MATLAB stage with argument values supplied in a temporary JSON job."""
import json
import os
from pathlib import Path
import subprocess
import uuid


def run_matlab(executable, function, job, output):
    if function not in ("beamform_rf", "batch_denoise", "evaluate_batch"):
        raise ValueError("Unknown MATLAB stage")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    job_file = output / ("job_" + uuid.uuid4().hex + ".json")
    job_file.write_text(json.dumps(job), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    quote = lambda text: "'" + str(text).replace("'", "''") + "'"
    expression = ";".join("addpath(" + quote(root / folder) + ")" for folder in ("imaging", "block_ksvd", "evaluation"))
    expression += ";" + function + "(" + quote(job_file) + ");"
    try:
        subprocess.run([os.fspath(executable), "-batch", expression], cwd=root, check=True)
    finally:
        job_file.unlink(missing_ok=True)
