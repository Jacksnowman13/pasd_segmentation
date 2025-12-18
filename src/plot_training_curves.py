import os
import csv
import sys
import matplotlib.pyplot as plt


def read_loss_csv(path):
    e, l = [], []
    f = open(path, "r")
    r = csv.reader(f)
    next(r, None)
    for row in r:
        if len(row) >= 2:
            e.append(int(row[0]))
            l.append(float(row[1]))
    f.close()
    return e, l


def try_plot(logs_dir, model_type, label):
    p = os.path.join(logs_dir, model_type + "_loss.csv")
    if not os.path.exists(p):
        return False
    e, l = read_loss_csv(p)
    plt.plot(e, l, label=label)
    return True


# AI assisted
def main():
    logs_dir = sys.argv[1]
    out_name = sys.argv[2]

    plt.figure()

    any_plotted = False
    any_plotted |= try_plot(logs_dir, "fair_cnn", "UPerNet-ConvNeXt-T (fair_cnn)")
    any_plotted |= try_plot(logs_dir, "fair_vit", "UPerNet-Swin-T (fair_vit)")

    if not any_plotted:
        raise FileNotFoundError("no loss csvs, somethings wrong ben")

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
