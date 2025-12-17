import os
import csv
import argparse
import matplotlib.pyplot as plt


def read_loss_csv(path):
    epochs, losses = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            epochs.append(int(row[0]))
            losses.append(float(row[1]))
    return epochs, losses


def try_plot(logs_dir, model_type, label=None):
    path = os.path.join(logs_dir, f"{model_type}_loss.csv")
    if not os.path.exists(path):
        return False
    epochs, losses = read_loss_csv(path)
    plt.plot(epochs, losses, label=label or model_type)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=r"..\logs")
    parser.add_argument("--out_name", type=str, default="training_loss_all_models.png")
    args = parser.parse_args()

    plt.figure()

    any_plotted = False
    any_plotted |= try_plot(args.logs_dir, "segformer", "SegFormer")
    any_plotted |= try_plot(args.logs_dir, "deeplab", "DeepLabv3-ResNet50")
    any_plotted |= try_plot(args.logs_dir, "fair_cnn", "UPerNet-ConvNeXt-T (fair_cnn)")
    any_plotted |= try_plot(args.logs_dir, "fair_vit", "UPerNet-Swin-T (fair_vit)")

    if not any_plotted:
        raise FileNotFoundError(f"No loss CSVs found in {args.logs_dir}")

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", args.out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve plot to {out_path}")


if __name__ == "__main__":
    main()
