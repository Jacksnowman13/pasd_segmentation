import os
import csv
import argparse
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "model_type": r["model_type"],
                "occ_type": r["occ_type"],
                "severity": float(r["severity"]),
                "clean_miou": float(r["clean_miou"]),
                "occ_miou": float(r["occ_miou"]),
                "delta_miou": float(r["delta_miou"]),
                "seed": int(r.get("seed", -1)),
            })
    return rows


def plot_from_rows(rows, metric, out_png, title):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["model_type"], []).append(r)

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda x: x["severity"])

    plt.figure()
    for model_type, rs in grouped.items():
        xs = [r["severity"] for r in rs]
        ys = [r[metric] for r in rs]
        plt.plot(xs, ys, marker="o", label=model_type)

    plt.xlabel("Occlusion severity (fraction)")
    plt.ylabel("Occluded mIoU" if metric == "occ_miou" else "ΔmIoU (clean - occluded)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["single", "all"], default="all",
                        help="single: plot one CSV; all: plot box/line/random and both metrics")
    parser.add_argument("--csv", type=str, default=None, help="used in single mode")
    parser.add_argument("--metric", type=str, choices=["occ_miou", "delta_miou"], default="occ_miou",
                        help="used in single mode")
    parser.add_argument("--csv_dir", type=str, default=r"..\eval_outputs", help="used in all mode")
    parser.add_argument("--out_dir", type=str, default=r"..\eval_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "single":
        if args.csv is None:
            raise ValueError("--csv is required in single mode")

        rows = read_csv(args.csv)
        occ_type = rows[0]["occ_type"] if rows else "unknown"
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_png = os.path.join(args.out_dir, f"{base}_{args.metric}.png")
        plot_from_rows(rows, args.metric, out_png, f"{occ_type} occlusion ({args.metric})")
        print("Saved:", out_png)
        return

    for occ in ["box", "line", "random"]:
        csv_path = os.path.join(args.csv_dir, f"occlusion_sweep_{occ}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}. Run sweep_occlusion.py first.")

        rows = read_csv(csv_path)

        out1 = os.path.join(args.out_dir, f"occlusion_{occ}_occ_miou.png")
        out2 = os.path.join(args.out_dir, f"occlusion_{occ}_delta_miou.png")

        plot_from_rows(rows, "occ_miou", out1, f"{occ} occlusion: occluded mIoU vs severity")
        plot_from_rows(rows, "delta_miou", out2, f"{occ} occlusion: ΔmIoU vs severity")

        print("Saved:", out1)
        print("Saved:", out2)

    print("\nDone. You should now have 6 plots in:", args.out_dir)


if __name__ == "__main__":
    main()
