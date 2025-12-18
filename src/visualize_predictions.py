import os
import sys
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


def voc_color_map(n):
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c & 1) << (7 - j)); c >>= 1
            g |= ((c & 1) << (7 - j)); c >>= 1
            b |= ((c & 1) << (7 - j)); c >>= 1
        cmap[i] = [r, g, b]
    return cmap


def colorize_mask(mask, cmap):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ok = (mask >= 0) & (mask < cmap.shape[0])
    out[ok] = cmap[mask[ok]]
    return out

# AI assisted
def main():
    MODEL_TYPE  = sys.argv[1]
    CHECKPOINT  = sys.argv[2]
    NUM_CLASSES = int(sys.argv[3])
    NUM_IMAGES  = int(sys.argv[4])
    VAL_IMG_DIR = sys.argv[5]
    VAL_MASK_DIR = sys.argv[6]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = A.Compose([A.Resize(512, 512)])
    ds = PASDDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=tfm)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    if MODEL_TYPE == "fair_cnn":
        model = load_fair_cnn(NUM_CLASSES)
    else:
        model = load_fair_vit(NUM_CLASSES)  

    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.to(device)
    model.eval()

    cmap = voc_color_map(NUM_CLASSES)

    out_dir = os.path.join(r"..\outputs", f"{MODEL_TYPE}_vis")
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        if MODEL_TYPE == "fair_cnn" or MODEL_TYPE == "fair_vit":
            logits = model(pixel_values=imgs).logits
        else:
            logits = model(imgs)["out"]

        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
        gt_np = masks[0].cpu().numpy()
        pred_np = preds[0].cpu().numpy()

        gt_color = colorize_mask(gt_np, cmap)
        pred_color = colorize_mask(pred_np, cmap)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1); plt.imshow(img_np);
        plt.title("Image"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(gt_color);
        plt.title("Ground truth"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(pred_color);
        plt.title("Prediction"); plt.axis("off")
        plt.tight_layout()

        out_path = os.path.join(out_dir, "example_" + str(count).zfill(3) + ".png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        count += 1
        if count >= NUM_IMAGES:
            break


if __name__ == "__main__":
    main()
