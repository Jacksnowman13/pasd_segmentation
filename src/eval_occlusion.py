import os
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A

from dataset import PASDDataset
from models_segformer import load_segformer
from models_deeplab import load_deeplab
from models_fair import load_fair_cnn, load_fair_vit


IGNORE_INDEX = 255

#Contains legacy code evaluating the previous models I was training and testing

def get_dataloader(img_dir, mask_dir, batch_size):
    transform = A.Compose([
        A.Resize(512, 512),
    ])
    ds = PASDDataset(img_dir, mask_dir, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # Was initally 0, adjusted to 4 to speed up training
        pin_memory=True,
        persistent_workers=True
    )
    return loader


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


def apply_random_cutout(imgs, cutout_frac=0.20, n_holes=8, value=0.0):
    b, c, h, w = imgs.shape
    side = max(1, int(min(h, w) * cutout_frac))

    out = imgs.clone()
    for i in range(b):
        for _ in range(n_holes):
            y0 = random.randint(0, max(0, h - side))
            x0 = random.randint(0, max(0, w - side))
            out[i, :, y0:y0 + side, x0:x0 + side] = value
    return out


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
    iou = tp / torch.clamp(denom, min=1)
    return iou


def load_model(model_type, num_classes, ckpt_path, device):
    if model_type == "segformer":
        model = load_segformer(num_classes)
    elif model_type == "deeplab":
        model = load_deeplab(num_classes)
    elif model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    elif model_type == "fair_vit":
        model = load_fair_vit(num_classes)
    else:
        raise ValueError("ERROR BEN!")

    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError("CHECKPOINT ERROR BEN")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_logits(model, imgs, model_type):
    hf_models = ["segformer", "fair_cnn", "fair_vit"]
    if model_type in hf_models:
        outputs = model(pixel_values=imgs)
        return outputs.logits
    elif model_type == "deeplab":
        outputs = model(imgs)
        return outputs["out"]
    else:
        raise ValueError("WHAT IS THIS MODEL BEN")


@torch.no_grad()
def run_occlusion_eval(model, loader, device, model_type, num_classes, occ_type, severity):
    clean_conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    occ_conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for imgs, masks in tqdm(loader, desc=f"Occlusion eval ({occ_type})", leave=True):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits_clean = infer_logits(model, imgs, model_type)
        logits_clean = F.interpolate(logits_clean, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds_clean = torch.argmax(logits_clean, dim=1).long()

        clean_conf = update_confusion_matrix(clean_conf, preds_clean.cpu(), masks.cpu(), num_classes)

        if occ_type == "box":
            imgs_occ = apply_box_occlusion(imgs, box_frac=severity, value=0.0)
        elif occ_type == "line":
            imgs_occ = apply_line_occlusion(imgs, band_frac=severity, value=0.0)
        elif occ_type == "random":
            cutout_frac = min(0.35, max(0.05, severity))
            n_holes = int(4 + 16 * severity)
            imgs_occ = apply_random_cutout(imgs, cutout_frac=cutout_frac, n_holes=n_holes, value=0.0)

        logits_occ = infer_logits(model, imgs_occ, model_type)
        logits_occ = F.interpolate(logits_occ, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds_occ = torch.argmax(logits_occ, dim=1).long()

        occ_conf = update_confusion_matrix(occ_conf, preds_occ.cpu(), masks.cpu(), num_classes)

    clean_iou = compute_iou_from_confmat(clean_conf.float())
    occ_iou = compute_iou_from_confmat(occ_conf.float())
    clean_miou = clean_iou.mean().item()
    occ_miou = occ_iou.mean().item()
    delta = clean_miou - occ_miou

    return clean_iou, occ_iou, clean_miou, occ_miou, delta

#AI assisted with argument parsing
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default=r"..\data\images_val")
    p.add_argument("--mask_dir", default=r"..\data\masks_val")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--model_type", choices=("segformer", "deeplab", "fair_cnn", "fair_vit"), required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--occ_type", choices=("box", "line", "random"), default="box")
    p.add_argument("--severity", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={a.model_type} ckpt={a.ckpt}")
    print(f"occ={a.occ_type} sev={a.severity} seed={a.seed}")

    loader = get_dataloader(a.img_dir, a.mask_dir, a.batch_size)
    model = load_model(a.model_type, a.num_classes, a.ckpt, device)

    clean_iou, occ_iou, clean_miou, occ_miou, delta = run_occlusion_eval(
        model, loader, device, a.model_type, a.num_classes, a.occ_type, a.severity
    )

    print(f"\nClean mIoU={clean_miou:.4f}  Occluded mIoU={occ_miou:.4f}  ΔmIoU={delta:.4f}")
    for c in range(a.num_classes):
        print(f"class {c:02d}: {clean_iou[c].item():.4f}->{occ_iou[c].item():.4f}")
    #AI assisted with saving the output
    out_dir = r"..\eval_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{a.model_type}_{a.occ_type}_sev{a.severity:.2f}.pt")
    torch.save(
        {
            "clean_iou": clean_iou.cpu(),
            "occ_iou": occ_iou.cpu(),
            "clean_miou": clean_miou,
            "occ_miou": occ_miou,
            "delta_miou": delta,
            "occ_type": a.occ_type,
            "severity": a.severity,
            "seed": a.seed,
        },
        out_path,
    )
    print("saved:", out_path)


if __name__ == "__main__":
    main()