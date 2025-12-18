import os
import sys
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
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c & 1) << (7 - j)); c >>= 1
            g |= ((c & 1) << (7 - j)); c >>= 1
            b |= ((c & 1) << (7 - j)); c >>= 1
        cmap[i] = [r, g, b]
    return cmap


def colorize_mask(mask, cmap):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    ok = (mask >= 0) & (mask < cmap.shape[0])
    out[ok] = cmap[mask[ok]]
    return out

#AI assisted
def main():
    VAL_IMG_DIR  = sys.argv[1]
    VAL_MASK_DIR = sys.argv[2]
    NUM_CLASSES  = int(sys.argv[3])

    MODEL_A_TYPE = sys.argv[4]
    MODEL_A_CKPT = sys.argv[5]
    LABEL_A      = sys.argv[6]

    MODEL_B_TYPE = sys.argv[7]
    MODEL_B_CKPT = sys.argv[8]
    LABEL_B      = sys.argv[9]

    NUM_EXAMPLES = int(sys.argv[10])
    SEED         = int(sys.argv[11])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PASDDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=A.Compose([A.Resize(512, 512)]))

    if MODEL_A_TYPE == "fair_cnn":
        mA = load_fair_cnn(NUM_CLASSES)
    else:
        mA = load_fair_vit(NUM_CLASSES)

    mA.load_state_dict(torch.load(MODEL_A_CKPT, map_location="cpu"))
    mA.to(device)
    mA.eval()

    if MODEL_B_TYPE == "fair_cnn":
        mB = load_fair_cnn(NUM_CLASSES)
    else:
        mB = load_fair_vit(NUM_CLASSES)

    mB.load_state_dict(torch.load(MODEL_B_CKPT, map_location="cpu"))
    mB.to(device)
    mB.eval()

    cmap = voc_color_map(NUM_CLASSES)

    random.seed(SEED)
    k = min(NUM_EXAMPLES, len(ds))
    idxs = random.sample(range(len(ds)), k)

    out_root = os.path.join("..", "outputs", "side_by_side", f"{MODEL_A_TYPE}_vs_{MODEL_B_TYPE}")
    os.makedirs(out_root, exist_ok=True)

    for r, idx in enumerate(idxs):
        img_t, mask_t = ds[idx]
        imgs = img_t.unsqueeze(0).to(device)
        masks = mask_t.unsqueeze(0).to(device)

        with torch.no_grad():
            if MODEL_A_TYPE == "fair_cnn" or MODEL_A_TYPE == "fair_vit":
                logitsA = mA(pixel_values=imgs).logits
            else:
                logitsA = mA(imgs)["out"]

            if MODEL_B_TYPE == "fair_cnn" or MODEL_B_TYPE == "fair_vit":
                logitsB = mB(pixel_values=imgs).logits
            else:
                logitsB = mB(imgs)["out"]

            logitsA = F.interpolate(logitsA, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            logitsB = F.interpolate(logitsB, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            pa = torch.argmax(logitsA, dim=1)[0].cpu()
            pb = torch.argmax(logitsB, dim=1)[0].cpu()

        img = img_t.numpy().transpose(1, 2, 0)
        gt_color = colorize_mask(mask_t.numpy(), cmap)
        a_color  = colorize_mask(pa.numpy(), cmap)
        b_color  = colorize_mask(pb.numpy(), cmap)

        plt.figure(figsize=(11, 3))
        plt.subplot(1, 4, 1); plt.imshow(img)
        plt.axis("off"); plt.title("Image")
        plt.subplot(1, 4, 2); plt.imshow(gt_color)
        plt.axis("off"); plt.title("Ground Truth")
        plt.subplot(1, 4, 3); plt.imshow(a_color)
        plt.axis("off"); plt.title(LABEL_A)
        plt.subplot(1, 4, 4); plt.imshow(b_color)
        plt.axis("off"); plt.title(LABEL_B)

        out_path = os.path.join(out_root, f"example_{r}_idx{idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
