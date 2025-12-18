import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from plot_class_frequency import count_labels, counts_to_array, VOC_CLASS_NAMES
from metrics import compute_iou_from_confusion


def load_iou(confmat_path):
    hist = torch.load(confmat_path, map_location="cpu")
    iou, miou = compute_iou_from_confusion(hist)
    return iou.numpy(), miou


# AI assisted
def main():
    confmat_ref = sys.argv[1]
    label_ref = sys.argv[2]
    confmat_cmp = sys.argv[3]
    label_cmp = sys.argv[4]
    out_name = sys.argv[5]

    val_img_dir = r"..\data\images_val"
    val_mask_dir = r"..\data\masks_val"
    counts = count_labels(val_img_dir, val_mask_dir)
    freq = counts_to_array(counts).astype(np.float32)
    freq[freq == 0] = 1.0
    log_freq = np.log10(freq)

    iou_ref, _ = load_iou(confmat_ref)
    iou_cmp, _ = load_iou(confmat_cmp)
    diff = iou_cmp - iou_ref

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(VOC_CLASS_NAMES):
        plt.scatter(log_freq[i], diff[i])
        plt.text(log_freq[i] + 0.01, diff[i], name, fontsize=8)

    plt.xlabel("log10(class pixel count in val)")
    plt.ylabel(f"ΔIoU ({label_cmp} − {label_ref})")
    plt.title("Class size vs IoU difference")
    plt.grid(True)

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
