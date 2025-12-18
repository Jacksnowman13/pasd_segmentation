import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import ConfusionMatrixDisplay

from dataset import PASDDataset
from models_fair import load_fair_cnn, load_fair_vit
from metrics import compute_confusion_matrix


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


def get_val_loader(val_img_dir, val_mask_dir, batch_size=2):
    transform = A.Compose([A.Resize(512, 512)])
    val_ds = PASDDataset(val_img_dir, val_mask_dir, transform=transform)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def compute_model_confusion(model, val_loader, device, num_classes):
    preds_list = []
    labels_list = []

    for imgs, masks in tqdm(val_loader, desc="Confusion"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        preds_list.extend(preds.cpu())
        labels_list.extend(masks.cpu())

    return compute_confusion_matrix(preds_list, labels_list, num_classes)


# Also, didn't know how to use sklearn very well, AI helped with this as well
def plot_cm_sklearn(cm, class_names, title, out_path, values_format=None, figsize=(10, 9)):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, xticks_rotation=45, cmap="viridis", values_format=values_format, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# AI assisted
def main():
    val_img_dir = sys.argv[1]
    val_mask_dir = sys.argv[2]
    num_classes = int(sys.argv[3])
    model_type = sys.argv[4]
    checkpoint = sys.argv[5]
    batch_size = int(sys.argv[6])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    val_loader = get_val_loader(val_img_dir, val_mask_dir, batch_size)

    if model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    else:
        model = load_fair_vit(num_classes)

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    hist = compute_model_confusion(model, val_loader, device, num_classes)
    cm = hist.numpy().astype(np.float32)

    figures_dir = os.path.join("..", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    base = "confusion_" + model_type

    plot_cm_sklearn(
        cm,
        VOC_CLASS_NAMES,
        title=f"Confusion Matrix (raw) - {model_type}",
        out_path=os.path.join(figures_dir, base + "_raw.png"),
        values_format=None,
    )

    cm_log = np.log1p(cm)
    plot_cm_sklearn(
        cm_log,
        VOC_CLASS_NAMES,
        title=f"Confusion Matrix (log1p raw) - {model_type}",
        out_path=os.path.join(figures_dir, base + "_log.png"),
        values_format=".2f",
    )

    row_sums = cm.sum(axis=1, keepdims=True) + 1e-6
    cm_norm = cm / row_sums
    plot_cm_sklearn(
        cm_norm,
        VOC_CLASS_NAMES,
        title=f"Confusion Matrix (row-normalized) - {model_type}",
        out_path=os.path.join(figures_dir, base + "_normalized.png"),
        values_format=".2f",
    )


if __name__ == "__main__":
    main()
