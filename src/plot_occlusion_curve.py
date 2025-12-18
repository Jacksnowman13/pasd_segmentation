import os
import sys
import csv
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    f = open(path, "r", newline="")
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
    f.close()
    return rows


def plot_from_rows(rows, metric, out_png, title):
    grouped = {}
    for r in rows:
        if r["model_type"] not in grouped:
            grouped[r["model_type"]] = []
        grouped[r["model_type"]].append(r)

    for k in grouped:
        grouped[k].sort(key=lambda x: x["severity"])

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


# AI assisted
def main():
    mode = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    if mode == "single":
        csv_path = sys.argv[3]
        metric = sys.argv[4]

        rows = read_csv(csv_path)
        occ_type = rows[0]["occ_type"] if rows else "unknown"
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = os.path.join(out_dir, base + "_" + metric + ".png")
        plot_from_rows(rows, metric, out_png, f"{occ_type} occlusion ({metric})")
        return

    csv_dir = sys.argv[3]

    for occ in ["box", "line", "random"]:
        csv_path = os.path.join(csv_dir, "occlusion_sweep_" + occ + ".csv")
        rows = read_csv(csv_path)

        out1 = os.path.join(out_dir, "occlusion_" + occ + "_occ_miou.png")
        out2 = os.path.join(out_dir, "occlusion_" + occ + "_delta_miou.png")

        plot_from_rows(rows, "occ_miou", out1, f"{occ} occlusion: occluded mIoU vs severity")
        plot_from_rows(rows, "delta_miou", out2, f"{occ} occlusion: ΔmIoU vs severity")


if __name__ == "__main__":
    main()
