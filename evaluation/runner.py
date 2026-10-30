"""Resolve evaluation cases before launching MATLAB and optional LPIPS."""
import csv
import math
import re
from common.files import contained_file, new_output
from common.matlab import run_matlab


def evaluate(settings, manifest, data_root, output_root, matlab, perceptual, device):
    required = ["sample_id", "group", "reference", "result", "scale_map", "rois", "tu", "mask"]
    cases = []
    with open(manifest, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != required:
            raise ValueError("Invalid evaluation manifest columns")
        for row in reader:
            if not re.fullmatch(r"sample_[0-9]{6}", row["sample_id"]) or not re.fullmatch(r"[a-z][a-z0-9_]*", row["group"]):
                raise ValueError("Use neutral sample and group identifiers")
            for key in required[2:]:
                if row[key]:
                    row[key] = str(contained_file(data_root, row[key]))
                elif key in ("reference", "result"):
                    raise ValueError("Every case needs a result and reference")
            cases.append(row)
    if not cases:
        raise ValueError("No evaluation cases")
    output = new_output(output_root)
    run_matlab(matlab, "evaluate_batch", {"parameters": settings, "cases": cases, "output": str(output)}, output)
    if perceptual:
        from .perceptual import perceptual_distance
        with (output / "metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields, records = reader.fieldnames, list(reader)
        for record, case in zip(records, cases):
            record["lpips"] = perceptual_distance(case["result"], case["reference"], settings["evaluation"]["lpips_network"], device)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        # Regenerate group means after adding perceptual distances.
        groups = {}
        for record in records:
            groups.setdefault(record["group"], []).append(record)
        with (output / "group_means.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["group", "count", *fields[2:]])
            for group, rows in groups.items():
                means = []
                for field in fields[2:]:
                    values = [float(row[field]) for row in rows if not math.isnan(float(row[field]))]
                    means.append(sum(values) / len(values) if values else float("nan"))
                writer.writerow([group, len(rows), *means])
    return output
