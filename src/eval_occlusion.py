import os
import sys
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

from dataset import PASDDataset
from models_fair import load_fair_cnn, load_fair_vit


IGNORE_INDEX = 255


def get_dataloader(img_dir, mask_dir, batch_size):
    transform = A.Compose([A.Resize(512, 512)])
    ds = PASDDataset(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def apply_box_occlusion(imgs, box_frac=0.25, value=0.0):
    b, c, h, w = imgs.shape
    side = max(1, int(min(h, w) * box_frac))
    out = imgs.clone()
    for i in range(b):
        y0 = random.randint(0, max(0, h - side))
        x0 = random.randint(0, max(0, w - side))
        out[i, :, y0:y0 + side, x0:x0 + side] = value
    return out


def apply_line_occlusion(imgs, band_frac=0.12, value=0.0):
    b, c, h, w = imgs.shape
    band = max(1, int(h * band_frac))
    out = imgs.clone()
    for i in range(b):
        y0 = random.randint(0, max(0, h - band))
        out[i, :, y0:y0 + band, :] = value
    return out


def apply_random_pixels(imgs, pixel_frac=0.20, value=0.0):
    b, c, h, w = imgs.shape
    out = imgs.clone()
    mask = torch.rand(b, 1, h, w, device=imgs.device) < pixel_frac
    return out.masked_fill(mask, value)


@torch.no_grad()
def update_confusion_matrix(conf_mat, preds, targets, num_classes):
    valid = (targets != IGNORE_INDEX) & (targets >= 0) & (targets < num_classes)
    t = targets[valid].view(-1)
    p = preds[valid].view(-1)
    idx = t * num_classes + p
    conf_mat += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return conf_mat


def compute_iou_from_confmat(conf_mat):
    tp = torch.diag(conf_mat)
    fp = conf_mat.sum(dim=0) - tp
    fn = conf_mat.sum(dim=1) - tp
    denom = tp + fp + fn
    return tp / torch.clamp(denom, min=1)


@torch.no_grad()
def run_occlusion_eval(model, loader, device, num_classes, occ_type, severity):
    clean_conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    occ_conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for imgs, masks in tqdm(loader, desc=f"Occlusion eval ({occ_type})", leave=True):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1).long()
        clean_conf = update_confusion_matrix(clean_conf, preds.cpu(), masks.cpu(), num_classes)

        if occ_type == "box":
            imgs_occ = apply_box_occlusion(imgs, box_frac=severity, value=0.0)
        elif occ_type == "line":
            imgs_occ = apply_line_occlusion(imgs, band_frac=severity, value=0.0)
        else:
            imgs_occ = apply_random_pixels(imgs, pixel_frac=severity, value=0.0)

        logits2 = model(pixel_values=imgs_occ).logits
        logits2 = F.interpolate(logits2, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds2 = torch.argmax(logits2, dim=1).long()
        occ_conf = update_confusion_matrix(occ_conf, preds2.cpu(), masks.cpu(), num_classes)

    clean_iou = compute_iou_from_confmat(clean_conf.float())
    occ_iou = compute_iou_from_confmat(occ_conf.float())
    clean_miou = clean_iou.mean().item()
    occ_miou = occ_iou.mean().item()
    delta = clean_miou - occ_miou
    return clean_iou, occ_iou, clean_miou, occ_miou, delta


# AI assisted
def main():
    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    num_classes = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    model_type = sys.argv[5]
    ckpt = sys.argv[6]
    occ_type = sys.argv[7]
    severity = float(sys.argv[8])
    seed = int(sys.argv[9])

    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={model_type} ckpt={ckpt}")
    print(f"occ={occ_type} sev={severity} seed={seed}")

    loader = get_dataloader(img_dir, mask_dir, batch_size)

    if model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    else:
        model = load_fair_vit(num_classes)

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    model.eval()

    clean_iou, occ_iou, clean_miou, occ_miou, delta = run_occlusion_eval(
        model, loader, device, num_classes, occ_type, severity
    )

    print(f"\nClean mIoU={clean_miou:.4f}  Occluded mIoU={occ_miou:.4f}  ΔmIoU={delta:.4f}")
    for c in range(num_classes):
        print(f"class {c:02d}: {clean_iou[c].item():.4f}->{occ_iou[c].item():.4f}")

    out_dir = r"..\eval_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_type}_{occ_type}_sev{severity:.2f}.pt")
    torch.save(
        {
            "clean_iou": clean_iou.cpu(),
            "occ_iou": occ_iou.cpu(),
            "clean_miou": clean_miou,
            "occ_miou": occ_miou,
            "delta_miou": delta,
            "occ_type": occ_type,
            "severity": severity,
            "seed": seed,
        },
        out_path,
    )


if __name__ == "__main__":
    main()
