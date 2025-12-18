import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A

from dataset import PASDDataset
from models_fair import load_fair_cnn, load_fair_vit
from metrics import _fast_hist


def voc_color_map(num_classes=21):
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c & 1) << (7 - j)); c >>= 1
            g |= ((c & 1) << (7 - j)); c >>= 1
            b |= ((c & 1) << (7 - j)); c >>= 1
        cmap[i] = np.array([r, g, b], dtype=np.uint8)
    return cmap


def colorize_mask(mask, cmap):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < cmap.shape[0])
    color[valid] = cmap[mask[valid]]
    return color


def compute_image_miou(pred, label, num_classes):
    pred = pred.view(-1)
    label = label.view(-1)

    valid = (label >= 0) & (label < num_classes)
    if valid.sum().item() == 0:
        return 0.0

    hist = _fast_hist(pred[valid], label[valid], num_classes).float()
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp
    denom = tp + fp + fn

    valid_classes = denom > 0
    if not valid_classes.any():
        return 0.0

    iou = torch.zeros_like(tp)
    iou[valid_classes] = tp[valid_classes] / denom[valid_classes]
    return float(iou[valid_classes].mean().item())


def save_triptych(img_tensor, gt_mask, pred_mask, cmap, out_path, title=None):
    img = img_tensor.numpy().transpose(1, 2, 0)
    gt_color = colorize_mask(gt_mask.numpy(), cmap)
    pred_color = colorize_mask(pred_mask.numpy(), cmap)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(img) 
    plt.axis("off"); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(gt_color)
    plt.axis("off"); plt.title("Ground truth")
    plt.subplot(1, 3, 3); plt.imshow(pred_color)
    plt.axis("off"); plt.title("Prediction")

    if title:
        plt.suptitle(title)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# AI assisted
def main():
    val_img_dir = sys.argv[1]
    val_mask_dir = sys.argv[2]
    num_classes = int(sys.argv[3])
    model_type = sys.argv[4]
    checkpoint = sys.argv[5]
    top_k = int(sys.argv[6])
    bottom_k = int(sys.argv[7])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = A.Compose([A.Resize(512, 512)])
    val_ds = PASDDataset(val_img_dir, val_mask_dir, transform=tfm)

    if model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    else:
        model = load_fair_vit(num_classes)  

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    cmap = voc_color_map(num_classes)

    scores = []
    for i in range(len(val_ds)):
        img_tensor, mask_tensor = val_ds[i]

        imgs = img_tensor.unsqueeze(0).to(device)
        masks = mask_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "fair_cnn" or model_type == "fair_vit":
                logits = model(pixel_values=imgs).logits
            else:
                logits = model(imgs)["out"]

            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)[0].cpu()

        miou = compute_image_miou(preds, mask_tensor.cpu(), num_classes)
        scores.append((i, miou))

    scores.sort(key=lambda x: x[1])
    bottom = scores[:bottom_k]
    top = scores[-top_k:]  # highest miou at end

    out_root = os.path.join("..", "outputs", f"{model_type}_best_worst")

    # worst
    for rank, (idx, miou) in enumerate(bottom):
        img_tensor, mask_tensor = val_ds[idx]

        imgs = img_tensor.unsqueeze(0).to(device)
        masks = mask_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "fair_cnn" or model_type == "fair_vit":
                logits = model(pixel_values=imgs).logits
            else:
                logits = model(imgs)["out"]
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)[0].cpu()

        out_path = os.path.join(out_root, f"worst_{rank}_idx{idx}_miou{miou:.3f}.png")
        save_triptych(img_tensor, mask_tensor, preds, cmap, out_path,
                      title=f"Worst {rank} (idx={idx}, mIoU={miou})")

    top = list(reversed(top))
    for rank, (idx, miou) in enumerate(top):
        img_tensor, mask_tensor = val_ds[idx]

        imgs = img_tensor.unsqueeze(0).to(device)
        masks = mask_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "fair_cnn" or model_type == "fair_vit":
                logits = model(pixel_values=imgs).logits
            else:
                logits = model(imgs)["out"]
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)[0].cpu()

        out_path = os.path.join(out_root, f"best_{rank}_idx{idx}_miou{miou:.3f}.png")
        save_triptych(img_tensor, mask_tensor, preds, cmap, out_path,
                      title=f"Best {rank} (idx={idx}, mIoU={miou})")


if __name__ == "__main__":
    main()
