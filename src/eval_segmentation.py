import os
import argparse

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


def get_dataloader(img_dir, mask_dir, batch_size):
    transform = A.Compose([
        A.Resize(512, 512),
    ])
    ds = PASDDataset(img_dir, mask_dir, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,          
        pin_memory=True,
        persistent_workers=True
    )
    return loader


@torch.no_grad()
def update_confusion_matrix(conf_mat, preds, targets, num_classes):
    """
    preds: [B,H,W] long
    targets: [B,H,W] long
    """
    valid = (targets != IGNORE_INDEX) & (targets >= 0) & (targets < num_classes)
    t = targets[valid].view(-1)
    p = preds[valid].view(-1)
    idx = t * num_classes + p
    conf_mat += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return conf_mat


def compute_iou_from_confmat(conf_mat):
    """
    conf_mat[i,j] = count of GT=i predicted=j
    """
    tp = torch.diag(conf_mat)
    fp = conf_mat.sum(dim=0) - tp
    fn = conf_mat.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = tp / torch.clamp(denom, min=1)
    return iou


def pixel_accuracy_from_confmat(conf_mat):
    correct = torch.diag(conf_mat).sum()
    total = conf_mat.sum()
    return (correct / torch.clamp(total, min=1)).item()


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
        raise ValueError(f"Unknown model_type: {model_type}")

    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_eval(model, loader, device, model_type, num_classes):
    hf_models = ["segformer", "fair_cnn", "fair_vit"]
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    for imgs, masks in tqdm(loader, desc="Evaluating", leave=True):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if model_type in hf_models:
            outputs = model(pixel_values=imgs)
            logits = outputs.logits
        elif model_type == "deeplab":
            outputs = model(imgs)
            logits = outputs["out"]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1).long()

        conf_mat = update_confusion_matrix(conf_mat, preds.cpu(), masks.cpu(), num_classes)

    iou = compute_iou_from_confmat(conf_mat.float())
    miou = iou.mean().item()
    pacc = pixel_accuracy_from_confmat(conf_mat.float())

    return conf_mat, iou.cpu(), miou, pacc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=r"..\data\images_val")
    parser.add_argument("--mask_dir", type=str, default=r"..\data\masks_val")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_type", type=str, choices=["segformer", "deeplab", "fair_cnn", "fair_vit"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Model:", args.model_type)
    print("Checkpoint:", args.ckpt)

    loader = get_dataloader(args.img_dir, args.mask_dir, args.batch_size)
    model = load_model(args.model_type, args.num_classes, args.ckpt, device)

    conf_mat, iou, miou, pacc = run_eval(model, loader, device, args.model_type, args.num_classes)

    print("\n=== Results ===")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Acc: {pacc:.4f}")

    print("\nPer-class IoU:")
    for c in range(args.num_classes):
        print(f"  class {c:02d}: {iou[c].item():.4f}")

    os.makedirs(r"..\eval_outputs", exist_ok=True)
    out_path = os.path.join(r"..\eval_outputs", f"{args.model_type}_confmat.pt")
    torch.save(conf_mat, out_path)
    print(f"\nSaved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
