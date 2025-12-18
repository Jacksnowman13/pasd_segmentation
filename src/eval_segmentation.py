import os
import sys

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


def pixel_accuracy_from_confmat(conf_mat):
    correct = torch.diag(conf_mat).sum()
    total = conf_mat.sum()
    return (correct / torch.clamp(total, min=1)).item()


@torch.no_grad()
def run_eval(model, loader, device, num_classes):
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for imgs, masks in tqdm(loader, desc="Evaluating", leave=True):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1).long()

        conf_mat = update_confusion_matrix(conf_mat, preds.cpu(), masks.cpu(), num_classes)

    iou = compute_iou_from_confmat(conf_mat.float())
    miou = iou.mean().item()
    pacc = pixel_accuracy_from_confmat(conf_mat.float())

    return conf_mat, iou.cpu(), miou, pacc


# AI assisted
def main():
    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    num_classes = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    model_type = sys.argv[5]
    ckpt = sys.argv[6]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_dataloader(img_dir, mask_dir, batch_size)

    if model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    else:
        model = load_fair_vit(num_classes)

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    model.eval()

    conf_mat, iou, miou, pacc = run_eval(model, loader, device, num_classes)

    print(f"Mean IoU: {miou:}, Pixel Accuracy: {pacc:}")

    os.makedirs(r"..\eval_outputs", exist_ok=True)
    out_path = os.path.join(r"..\eval_outputs", model_type + "_confmat.pt")
    torch.save(conf_mat, out_path)


if __name__ == "__main__":
    main()
