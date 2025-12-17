import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A

from dataset import PASDDataset
from models_segformer import load_segformer
from models_deeplab import load_deeplab
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
        cmap[i] = np.array([r, g, b])
    return cmap


def colorize_mask(mask, cmap):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < cmap.shape[0])
    color[valid] = cmap[mask[valid]]
    return color


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


@torch.no_grad()
def predict_mask(model, img_tensor, mask_tensor, device, model_type):
    hf_models = ["segformer", "fair_cnn", "fair_vit"]

    imgs = img_tensor.unsqueeze(0).to(device)
    masks = mask_tensor.unsqueeze(0).to(device)

    if model_type in hf_models:
        logits = model(pixel_values=imgs).logits
    else:
        logits = model(imgs)["out"]

    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    preds = torch.argmax(logits, dim=1)[0].cpu()
    return preds


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
    plt.subplot(1, 3, 1); plt.imshow(img); plt.axis("off"); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(gt_color); plt.axis("off"); plt.title("Ground truth")
    plt.subplot(1, 3, 3); plt.imshow(pred_color); plt.axis("off"); plt.title("Prediction")

    if title:
        plt.suptitle(title)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_img_dir", type=str, default=r"..\data\images_val")
    parser.add_argument("--val_mask_dir", type=str, default=r"..\data\masks_val")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--model_type", type=str, choices=["segformer", "deeplab", "fair_cnn", "fair_vit"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--bottom_k", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = A.Compose([A.Resize(512, 512)])
    val_ds = PASDDataset(args.val_img_dir, args.val_mask_dir, transform=transform)

    model = build_model(args.model_type, args.num_classes, args.checkpoint, device)
    cmap = voc_color_map(num_classes=args.num_classes)

    # compute per-image mIoU
    scores = []
    for idx in range(len(val_ds)):
        img_tensor, mask_tensor = val_ds[idx]
        preds = predict_mask(model, img_tensor, mask_tensor, device, args.model_type)
        miou = compute_image_miou(preds, mask_tensor.cpu(), args.num_classes)
        scores.append((idx, miou))

    scores.sort(key=lambda x: x[1])
    bottom = scores[:args.bottom_k]
    top = scores[-args.top_k:]

    print(f"Lowest {args.bottom_k} mIoU samples:", bottom)
    print(f"Highest {args.top_k} mIoU samples:", top)

    out_root = os.path.join("..", "outputs", f"{args.model_type}_best_worst")

    # Worst
    for rank, (idx, miou) in enumerate(bottom):
        img_tensor, mask_tensor = val_ds[idx]
        preds = predict_mask(model, img_tensor, mask_tensor, device, args.model_type)
        out_path = os.path.join(out_root, f"worst_{rank}_idx{idx}_miou{miou:.3f}.png")
        save_triptych(img_tensor, mask_tensor, preds, cmap, out_path, title=f"Worst {rank} (idx={idx}, mIoU={miou:.3f})")

    # Best
    for rank, (idx, miou) in enumerate(reversed(top)):
        img_tensor, mask_tensor = val_ds[idx]
        preds = predict_mask(model, img_tensor, mask_tensor, device, args.model_type)
        out_path = os.path.join(out_root, f"best_{rank}_idx{idx}_miou{miou:.3f}.png")
        save_triptych(img_tensor, mask_tensor, preds, cmap, out_path, title=f"Best {rank} (idx={idx}, mIoU={miou:.3f})")

    print(f"Saved best/worst examples under {out_root}")


if __name__ == "__main__":
    main()
