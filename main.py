"""
main.py

Convenience entry point for the DeepShieldIDS project.
Run with no arguments for an interactive menu, or pass a stage directly.

Examples
--------
python main.py --stage train --data data/raw/CIC-IDS2017.csv
python main.py --stage evaluate --data data/raw/CIC-IDS2017.csv --checkpoint models/hybridids_best.pt
python main.py --stage cross_dataset --dataset_a data/raw/cic.csv --dataset_b data/raw/unsw.csv
python main.py --stage compare --data data/raw/CIC-IDS2017.csv
python main.py --stage demo
"""

import sys
import argparse
import subprocess

STAGES = ["demo", "train", "evaluate", "cross_dataset", "compare"]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="DeepShieldIDS project runner")
    parser.add_argument("--stage", choices=STAGES, default=None,
                         help="Which pipeline stage to run.")
    # pass-through args
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--dataset_a", type=str, default=None)
    parser.add_argument("--dataset_b", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="models/hybridids_best.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tag", type=str, default="dataset")
    args, unknown = parser.parse_known_args()

    if args.stage is None:
        print("No --stage given. Running full demo pipeline on synthetic data...\n")
        args.stage = "demo"

    if args.stage == "demo":
        run([sys.executable, "src/train.py", "--epochs", str(args.epochs), "--tag", "demo"])
        run([sys.executable, "src/evaluate.py", "--tag", "demo",
             "--checkpoint", "models/hybridids_best.pt"])

    elif args.stage == "train":
        cmd = [sys.executable, "src/train.py", "--epochs", str(args.epochs),
               "--tag", args.tag, "--checkpoint", args.checkpoint]
        if args.data:
            cmd += ["--data", args.data]
        run(cmd)

    elif args.stage == "evaluate":
        cmd = [sys.executable, "src/evaluate.py", "--tag", args.tag,
               "--checkpoint", args.checkpoint]
        if args.data:
            cmd += ["--data", args.data]
        run(cmd)

    elif args.stage == "cross_dataset":
        cmd = [sys.executable, "src/cross_dataset_eval.py", "--epochs", str(args.epochs)]
        if args.dataset_a:
            cmd += ["--dataset_a", args.dataset_a]
        if args.dataset_b:
            cmd += ["--dataset_b", args.dataset_b]
        run(cmd)

    elif args.stage == "compare":
        cmd = [sys.executable, "src/compare_baselines.py", "--epochs", str(args.epochs),
               "--tag", args.tag]
        if args.data:
            cmd += ["--data", args.data]
        run(cmd)


if __name__ == "__main__":
    main()
