import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from plot_class_frequency import count_labels, counts_to_array, VOC_CLASS_NAMES
from metrics import compute_iou_from_confusion


def load_iou(confmat_path):
    hist = torch.load(confmat_path, map_location="cpu")
    iou, miou = compute_iou_from_confusion(hist)
    return iou.numpy(), miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confmat_ref", type=str, required=True)
    parser.add_argument("--label_ref", type=str, default="Ref")
    parser.add_argument("--confmat_cmp", type=str, required=True)
    parser.add_argument("--label_cmp", type=str, default="Cmp")
    parser.add_argument("--out_name", type=str, default="size_vs_delta_iou.png")
    args = parser.parse_args()

    # 1) Val class frequencies
    val_img_dir = r"..\data\images_val"
    val_mask_dir = r"..\data\masks_val"
    counts = count_labels(val_img_dir, val_mask_dir)
    freq = counts_to_array(counts).astype(np.float32)
    freq[freq == 0] = 1.0
    log_freq = np.log10(freq)

    # 2) IoU diff
    iou_ref, _ = load_iou(args.confmat_ref)
    iou_cmp, _ = load_iou(args.confmat_cmp)
    diff = iou_cmp - iou_ref

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(VOC_CLASS_NAMES):
        plt.scatter(log_freq[i], diff[i])
        plt.text(log_freq[i] + 0.01, diff[i], name, fontsize=8)

    plt.xlabel("log10(class pixel count in val)")
    plt.ylabel(f"ΔIoU ({args.label_cmp} − {args.label_ref})")
    plt.title("Class size vs IoU difference")
    plt.grid(True)

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", args.out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    main()
