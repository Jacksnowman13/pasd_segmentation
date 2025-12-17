import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A

from dataset import PASDDataset
from models_segformer import load_segformer
from models_deeplab import load_deeplab
from models_fair import load_fair_cnn, load_fair_vit


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
def forward_logits(model, imgs, masks, model_type):
    hf_models = ["segformer", "fair_cnn", "fair_vit"]

    if model_type in hf_models:
        logits = model(pixel_values=imgs).logits
    else:
        logits = model(imgs)["out"]

    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_img_dir", type=str, default=r"..\data\images_val")
    parser.add_argument("--val_mask_dir", type=str, default=r"..\data\masks_val")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["segformer", "deeplab", "fair_cnn", "fair_vit"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = A.Compose([A.Resize(512, 512)])
    dataset = PASDDataset(args.val_img_dir, args.val_mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = build_model(args.model_type, args.num_classes, args.checkpoint, device)
    cmap = voc_color_map(args.num_classes)

    out_dir = os.path.join(r"..\outputs", f"{args.model_type}_vis")
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = forward_logits(model, imgs, masks, args.model_type)
        preds = torch.argmax(logits, dim=1)

        img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
        gt_np = masks[0].cpu().numpy()
        pred_np = preds[0].cpu().numpy()

        gt_color = colorize_mask(gt_np, cmap)
        pred_color = colorize_mask(pred_np, cmap)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1); plt.imshow(img_np); plt.title("Image"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(gt_color); plt.title("Ground truth"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(pred_color); plt.title("Prediction"); plt.axis("off")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"example_{count:03d}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        count += 1
        if count >= args.num_images:
            break

    print("Saved visuals to:", out_dir)


if __name__ == "__main__":
    main()
