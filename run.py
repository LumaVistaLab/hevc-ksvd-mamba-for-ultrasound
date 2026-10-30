"""Run the ultrasound enhancement stages with an explicit parameter file."""
import argparse
from pathlib import Path
from common.config import load_config, save_config
from common.files import image_files, new_output
from common.matlab import run_matlab


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment parameters")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="Override a dotted parameter key")
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "prior", "beamform", "denoise", "fixed", "train", "infer", "evaluate"):
        command = stages.add_parser(name)
        command.add_argument("--output", required=True)
        if name in ("prepare", "prior", "beamform", "infer"):
            command.add_argument("--input", required=True)
        if name == "prior":
            command.add_argument("--encoder", required=True)
        if name in ("beamform", "denoise", "fixed", "evaluate"):
            command.add_argument("--matlab", required=True, help="MATLAB executable or command")
        if name == "beamform":
            command.add_argument("--beams", required=True, type=int)
        if name in ("denoise", "fixed"):
            command.add_argument("--images", required=True)
            command.add_argument("--profile", required=True)
        if name == "denoise":
            command.add_argument("--priors", required=True)
        if name in ("train", "evaluate"):
            command.add_argument("--manifest", required=True)
            command.add_argument("--data-root", required=True)
        if name in ("train", "infer", "evaluate"):
            command.add_argument("--device", required=True)
        if name == "train":
            command.add_argument("--variant", required=True, choices=("full", "conv_bottleneck", "no_fourier", "no_refinement", "raster_scan"))
        if name == "infer":
            command.add_argument("--checkpoint", required=True)
        if name == "evaluate":
            command.add_argument("--lpips", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    settings = load_config(args.config, args.set)
    if args.stage == "prepare":
        from imaging.preprocess import prepare
        prepare(args.input, args.output, settings["imaging"])
    elif args.stage == "prior":
        from hevc_prior.runner import extract
        extract(settings, args.encoder, args.input, args.output)
    elif args.stage == "beamform":
        if args.beams < 1:
            raise ValueError("Beamline count must be positive")
        source = Path(args.input).resolve(strict=True)
        output = new_output(args.output, source)
        run_matlab(args.matlab, "beamform_rf", {"parameters": settings, "input": str(source), "output": str(output), "beams": args.beams}, output)
    elif args.stage in ("denoise", "fixed"):
        images, _ = image_files(args.images)
        if args.profile not in settings["block_ksvd"]["profiles"]:
            raise ValueError("Unknown block-size profile")
        output = new_output(args.output, images)
        job = {"parameters": settings, "images": str(images), "output": str(output), "profile": args.profile, "fixed": args.stage == "fixed", "priors": ""}
        if args.stage == "denoise":
            job["priors"] = str(Path(args.priors).resolve(strict=True))
        run_matlab(args.matlab, "batch_denoise", job, output)
    elif args.stage == "train":
        from usr_mamba.training import train
        train(settings, args.manifest, args.data_root, args.output, args.variant, args.device)
    elif args.stage == "infer":
        from usr_mamba.inference import enhance
        enhance(args.checkpoint, args.input, args.output, args.device)
    elif args.stage == "evaluate":
        from evaluation.runner import evaluate
        evaluate(settings, args.manifest, args.data_root, args.output, args.matlab, args.lpips, args.device)
    if args.stage != "train":
        save_config(settings, Path(args.output) / "parameters.json")


if __name__ == "__main__":
    main()
