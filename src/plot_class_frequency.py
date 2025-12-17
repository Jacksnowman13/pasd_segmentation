import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from dataset import PASDDataset

VOC_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

def count_labels(img_dir, mask_dir):
    transform = A.Compose([A.Resize(512, 512)])
    ds = PASDDataset(img_dir, mask_dir, transform=transform)

    counts = Counter()
    for _, mask in ds:
        unique, cts = mask.unique(return_counts=True)
        for u, c in zip(unique.tolist(), cts.tolist()):
            counts[u] += c
    return counts

def counts_to_array(counts, num_classes=21):
    arr = np.zeros(num_classes, dtype=np.int64)
    for k, v in counts.items():
        if 0 <= k < num_classes:
            arr[k] = v
    return arr

def plot_counts(counts_arr, title, out_path):
    x = np.arange(len(VOC_CLASS_NAMES))
    plt.figure(figsize=(10, 4))
    plt.bar(x, counts_arr)
    plt.xticks(x, VOC_CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("Pixel count")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved class-frequency plot to {out_path}")

def main():
    train_img_dir = r"..\data\images_train"
    train_mask_dir = r"..\data\masks_train"
    val_img_dir = r"..\data\images_val"
    val_mask_dir = r"..\data\masks_val"

    train_counts = count_labels(train_img_dir, train_mask_dir)
    val_counts = count_labels(val_img_dir, val_mask_dir)

    train_arr = counts_to_array(train_counts)
    val_arr = counts_to_array(val_counts)

    fig_dir = r"..\figures"
    os.makedirs(fig_dir, exist_ok=True)

    plot_counts(train_arr, "Class Frequency (Train, pixels)", os.path.join(fig_dir, "class_freq_train.png"))
    plot_counts(val_arr, "Class Frequency (Val, pixels)", os.path.join(fig_dir, "class_freq_val.png"))

if __name__ == "__main__":
    main()
