import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import ConfusionMatrixDisplay

from dataset import PASDDataset
from models_segformer import load_segformer
from models_deeplab import load_deeplab
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


def build_model(model_type, num_classes, checkpoint_path, device):
    if model_type == "segformer":
        model = load_segformer(num_classes)
    elif model_type == "deeplab":
        model = load_deeplab(num_classes)
    elif model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    elif model_type == "fair_vit":
        model = load_fair_vit(num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_val_loader(val_img_dir, val_mask_dir, batch_size=2):
    transform = A.Compose([A.Resize(512, 512)])
    val_ds = PASDDataset(val_img_dir, val_mask_dir, transform=transform)
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


@torch.no_grad()
def compute_model_confusion(model, val_loader, device, num_classes, model_type):
    preds_list = []
    labels_list = []
    hf_models = ["segformer", "fair_cnn", "fair_vit"]

    for imgs, masks in tqdm(val_loader, desc="Computing confusion"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if model_type in hf_models:
            logits = model(pixel_values=imgs).logits
        else:
            logits = model(imgs)["out"]

        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        preds_list.extend(preds.cpu())
        labels_list.extend(masks.cpu())

    hist = compute_confusion_matrix(preds_list, labels_list, num_classes)
    return hist


def plot_cm_sklearn(cm, class_names, title, out_path, values_format=None, figsize=(10, 9)):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, xticks_rotation=45, cmap="viridis", values_format=values_format, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_img_dir", type=str, default=r"..\data\images_val")
    parser.add_argument("--val_mask_dir", type=str, default=r"..\data\masks_val")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--model_type", type=str, choices=["segformer", "deeplab", "fair_cnn", "fair_vit"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    val_loader = get_val_loader(args.val_img_dir, args.val_mask_dir, args.batch_size)
    model = build_model(args.model_type, args.num_classes, args.checkpoint, device)

    hist = compute_model_confusion(model, val_loader, device, args.num_classes, args.model_type)
    cm = hist.numpy().astype(np.float32)

    figures_dir = os.path.join("..", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    base = f"confusion_{args.model_type}"

    plot_cm_sklearn(
        cm,
        VOC_CLASS_NAMES,
        title=f"Confusion Matrix (raw) - {args.model_type}",
        out_path=os.path.join(figures_dir, base + "_raw.png"),
        values_format=None, 
    )

    cm_log = np.log1p(cm)
    plot_cm_sklearn(
        cm_log,
        VOC_CLASS_NAMES,
        title=f"Confusion Matrix (log1p raw) - {args.model_type}",
        out_path=os.path.join(figures_dir, base + "_log.png"),
        values_format=".2f",
    )

    row_sums = cm.sum(axis=1, keepdims=True) + 1e-6
    cm_norm = cm / row_sums
    plot_cm_sklearn(
        cm_norm,
        VOC_CLASS_NAMES,
        title=f"Confusion Matrix (row-normalized) - {args.model_type}",
        out_path=os.path.join(figures_dir, base + "_normalized.png"),
        values_format=".2f",
    )


if __name__ == "__main__":
    main()
