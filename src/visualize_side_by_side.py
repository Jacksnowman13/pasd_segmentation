import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A

from dataset import PASDDataset
from models_segformer import load_segformer
from models_deeplab import load_deeplab
from models_fair import load_fair_cnn, load_fair_vit

def voc_color_map(num_classes=21):
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):
        r = 0
        g = 0
        b = 0
        c = i
        for j in range(8):
            r |= ((c & 1) << (7 - j)); c >>= 1
            g |= ((c & 1) << (7 - j)); c >>= 1
            b |= ((c & 1) << (7 - j)); c >>= 1
        cmap[i] = np.array([r, g, b], dtype=np.uint8)

    return cmap


def colorize_mask(mask, cmap):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ok = (mask >= 0) & (mask < cmap.shape[0])
    out[ok] = cmap[mask[ok]]
    return out


def build_model(model_type, num_classes, checkpoint_path, device):
    m = None
    if model_type == "segformer":
        m = load_segformer(num_classes)
    elif model_type == "deeplab":
        m = load_deeplab(num_classes)
    elif model_type == "fair_cnn":
        m = load_fair_cnn(num_classes)
    elif model_type == "fair_vit":
        m = load_fair_vit(num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    sd = torch.load(checkpoint_path, map_location="cpu")
    m.load_state_dict(sd)
    m.to(device)
    m.eval()
    return m


@torch.no_grad()
def run_model(model, imgs, masks, model_type):
    hf_models = ["fair_cnn", "fair_vit"]

    if model_type in hf_models:
        logits = model(pixel_values=imgs).logits
    else:
        logits = model(imgs)["out"]

    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    preds = torch.argmax(logits, dim=1)
    return preds


def save_side_by_side(img_tensor, gt_mask, pred_a, pred_b, cmap, out_path, label_a="Model A", label_b="Model B"):
    img = img_tensor.numpy()
    img = img.transpose(1, 2, 0)

    gt_color = colorize_mask(gt_mask.numpy(), cmap)
    a_color = colorize_mask(pred_a.numpy(), cmap)
    b_color = colorize_mask(pred_b.numpy(), cmap)

    plt.figure(figsize=(11, 3))
    plt.subplot(1, 4, 1); plt.imshow(img);      plt.axis("off"); plt.title("Image")
    plt.subplot(1, 4, 2); plt.imshow(gt_color); plt.axis("off"); plt.title("Ground Truth")
    plt.subplot(1, 4, 3); plt.imshow(a_color);  plt.axis("off"); plt.title(label_a)
    plt.subplot(1, 4, 4); plt.imshow(b_color);  plt.axis("off"); plt.title(label_b)

    d = os.path.dirname(out_path)
    os.makedirs(d, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # AI assisted 
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_img_dir", type=str, default=r"..\data\images_val")
    parser.add_argument("--val_mask_dir", type=str, default=r"..\data\masks_val")
    parser.add_argument("--num_classes", type=int, default=21)

    parser.add_argument("--model_a_type", type=str, choices=["segformer", "deeplab", "fair_cnn", "fair_vit"], required=True)
    parser.add_argument("--model_a_ckpt", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="Model A")

    parser.add_argument("--model_b_type", type=str, choices=["segformer", "deeplab", "fair_cnn", "fair_vit"], required=True)
    parser.add_argument("--model_b_ckpt", type=str, required=True)
    parser.add_argument("--label_b", type=str, default="Model B")

    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    transform = A.Compose([A.Resize(512, 512)])
    ds = PASDDataset(args.val_img_dir, args.val_mask_dir, transform=transform)

    mA = build_model(args.model_a_type, args.num_classes, args.model_a_ckpt, device)
    mB = build_model(args.model_b_type, args.num_classes, args.model_b_ckpt, device)

    random.seed(args.seed)
    n = len(ds)
    k = args.num_examples
    if k > n:
        k = n
    idxs = random.sample(range(n), k=k)

    cmap = voc_color_map(num_classes=args.num_classes)
    out_root = os.path.join("..", "outputs", "side_by_side", f"{args.model_a_type}_vs_{args.model_b_type}")

    r = 0
    for idx in idxs:
        img_t, mask_t = ds[idx]

        imgs = img_t.unsqueeze(0).to(device)
        masks = mask_t.unsqueeze(0).to(device)

        pa = run_model(mA, imgs, masks, args.model_a_type)[0].cpu()
        pb = run_model(mB, imgs, masks, args.model_b_type)[0].cpu()

        out_path = os.path.join(out_root, f"example_{r}_idx{idx}.png")
        save_side_by_side(img_t, mask_t, pa, pb, cmap, out_path, args.label_a, args.label_b)
        r += 1


if __name__ == "__main__":
    main()
