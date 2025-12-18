import os
import csv
import argparse
import subprocess
from pathlib import Path

# Debugged with the help of ChatGPT
def run_one(model_type, ckpt, occ_type, severity, seed, num_classes, batch_size):
    cmd = [
        "python", "eval_occlusion.py",
        "--num_classes", str(num_classes),
        "--batch_size", str(batch_size),
        "--model_type", model_type,
        "--ckpt", ckpt,
        "--occ_type", occ_type,
        "--severity", f"{severity:.2f}",
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True)


def collect_rows(eval_dir, occ_type, severities):
    import torch

    rows = []
    for sev in severities:
        for model_type in ["fair_cnn", "fair_vit"]:
            pt_path = os.path.join(eval_dir, f"{model_type}_{occ_type}_sev{sev:.2f}.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Missing output: {pt_path} (did eval_occlusion.py save it?)")
            d = torch.load(pt_path, map_location="cpu")
            rows.append({
                "model_type": model_type,
                "occ_type": occ_type,
                "severity": float(sev),
                "clean_miou": float(d["clean_miou"]),
                "occ_miou": float(d["occ_miou"]),
                "delta_miou": float(d["delta_miou"]),
                "seed": int(d.get("seed", -1)),
            })

    rows = sorted(rows, key=lambda r: (r["model_type"], r["severity"]))
    return rows


def write_csv(path, rows):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_type", "occ_type", "severity", "clean_miou", "occ_miou", "delta_miou", "seed"],
        )
        writer.writeheader()
        writer.writerows(rows)

# AI assisted with argument parsing and folder creation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--end", type=float, default=0.35)
    parser.add_argument("--step", type=float, default=0.05)

    parser.add_argument("--ckpt_cnn", type=str, default=r"..\checkpoints\fair_cnn_best.pt")
    parser.add_argument("--ckpt_vit", type=str, default=r"..\checkpoints\fair_vit_best.pt")

    parser.add_argument("--eval_dir", type=str, default=r"..\eval_outputs")
    parser.add_argument("--out_dir", type=str, default=r"..\eval_outputs")
    args = parser.parse_args()

    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    severities = []
    s = args.start
    while s <= args.end + 1e-9:
        severities.append(round(s, 2))
        s += args.step

    occ_types = ["box", "line", "random"]

    for occ in occ_types:
        for sev in severities:
            run_one("fair_cnn", args.ckpt_cnn, occ, sev, args.seed, args.num_classes, args.batch_size)
            run_one("fair_vit", args.ckpt_vit, occ, sev, args.seed, args.num_classes, args.batch_size)

    for occ in occ_types:
        rows = collect_rows(args.eval_dir, occ, severities)
        out_csv = os.path.join(args.out_dir, f"occlusion_sweep_{occ}.csv")
        write_csv(out_csv, rows)
        print(f"\nSaved CSV: {out_csv}")



if __name__ == "__main__":
    main()
