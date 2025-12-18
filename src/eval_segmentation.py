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
    # AI help: adapted parts from eval_occlusion into this eval script.
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default=r"..\data\images_val")
    p.add_argument("--mask_dir", default=r"..\data\masks_val")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--model_type", choices=("segformer", "deeplab", "fair_cnn", "fair_vit"), required=True)
    p.add_argument("--ckpt", required=True)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_dataloader(a.img_dir, a.mask_dir, a.batch_size)
    model = load_model(a.model_type, a.num_classes, a.ckpt, device)

    conf_mat, iou, miou, pacc = run_eval(model, loader, device, a.model_type, a.num_classes)

    os.makedirs(r"..\eval_outputs", exist_ok=True)
    out_path = os.path.join(r"..\eval_outputs", a.model_type + "_confmat.pt")
    torch.save(conf_mat, out_path)


if __name__ == "__main__":
    main()
