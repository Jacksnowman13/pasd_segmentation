import os
import csv
import sys
import runpy
import torch


def run_eval_occlusion(num_classes, batch_size, seed, model_type, ckpt, occ_type, severity):
    sys.argv = [
        "eval_occlusion.py",
        "--num_classes", str(num_classes),
        "--batch_size", str(batch_size),
        "--model_type", model_type,
        "--ckpt", ckpt,
        "--occ_type", occ_type,
        "--severity", f"{severity:.2f}",
        "--seed", str(seed),
    ]
    runpy.run_path("eval_occlusion.py", run_name="__main__")


# AI assisted
def main():
    num_classes = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    seed = int(sys.argv[3])
    ckpt_cnn = sys.argv[4]
    ckpt_vit = sys.argv[5]
    eval_dir = sys.argv[6]
    out_dir = sys.argv[7]
    start = float(sys.argv[8])
    end = float(sys.argv[9])
    step = float(sys.argv[10])

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    occ_types = ["box", "line", "random"]
    models = [("fair_cnn", ckpt_cnn), ("fair_vit", ckpt_vit)]

    severities = []
    s = start
    while s <= end + 1e-9:
        severities.append(round(s, 2))
        s += step

    for occ in occ_types:
        for sev in severities:
            for model_type, ckpt in models:
                print("running:", model_type, occ, sev)
                try:
                    run_eval_occlusion(num_classes, batch_size, seed, model_type, ckpt, occ, sev)
                except Exception:
                    print("FAILED")

    fields = ["model_type", "occ_type", "severity", "clean_miou", "occ_miou", "delta_miou", "seed"]

    for occ in occ_types:
        rows = []
        for sev in severities:
            for model_type, _ in models:
                pt_path = os.path.join(eval_dir, f"{model_type}_{occ}_sev{sev:.2f}.pt")
                if not os.path.exists(pt_path):
                    print("missing:", pt_path)
                    continue
                d = torch.load(pt_path, map_location="cpu")
                rows.append({
                    "model_type": model_type,
                    "occ_type": occ,
                    "severity": float(sev),
                    "clean_miou": float(d["clean_miou"]),
                    "occ_miou": float(d["occ_miou"]),
                    "delta_miou": float(d["delta_miou"]),
                    "seed": int(d.get("seed", -1)),
                })

        rows.sort(key=lambda r: (r["model_type"], r["severity"]))

        out_csv = os.path.join(out_dir, f"occlusion_sweep_{occ}.csv")
        f = open(out_csv, "w", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        f.close()


if __name__ == "__main__":
    main()
