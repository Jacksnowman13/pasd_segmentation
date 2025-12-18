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


def get_dataloaders(train_img_dir, train_mask_dir,
                    val_img_dir, val_mask_dir,
                    batch_size):
    transform = A.Compose([
        A.Resize(512, 512),
    ])

    train_ds = PASDDataset(train_img_dir, train_mask_dir, transform=transform)
    val_ds = PASDDataset(val_img_dir, val_mask_dir, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, device, model_type, num_classes):
    model.train()
    total_loss = 0.0

    hf_models = ["segformer", "fair_cnn", "fair_vit"]

    for imgs, masks in tqdm(train_loader, desc="Training", leave=False):
        imgs = imgs.to(device, non_blocking = True)
        masks = masks.to(device, non_blocking = True)

        optimizer.zero_grad()

        if model_type in hf_models:
            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        elif model_type == "deeplab":
            outputs = model(imgs)
            logits = outputs["out"]
        else:
            raise ValueError("Uknown model type")

        targets = masks.clone().long()
        invalid = (targets < 0) | (targets >= num_classes)
        targets[invalid] = 255  

        loss = F.cross_entropy(logits, targets, ignore_index=255)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# AI assisted here with argument parsing and folder creation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_dir", type=str, default=r"..\data\images_train")
    parser.add_argument("--train_mask_dir", type=str, default=r"..\data\masks_train")
    parser.add_argument("--val_img_dir", type=str, default=r"..\data\images_val")
    parser.add_argument("--val_mask_dir", type=str, default=r"..\data\masks_val")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--model_type",
        type=str, #Legacy code from my previous iterations of this project
        choices=["segformer", "deeplab", "fair_cnn", "fair_vit"],
        default="segformer",
    )
    parser.add_argument("--save_path", type=str, default=r"..\checkpoints")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists(r"..\logs"):
        os.makedirs(r"..\logs", exist_ok=True)

    log_path = os.path.join(r"..\logs", f"{args.model_type}_loss.csv")
    f = open(log_path, "w")
    f.write("epoch,loss\n")
    f.close()

    train_loader, _ = get_dataloaders(
        args.train_img_dir,
        args.train_mask_dir,
        args.val_img_dir,
        args.val_mask_dir,
        args.batch_size,
    )

    model = None
    optimizer = None
    if args.model_type == "segformer":
        model = load_segformer(args.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.model_type == "deeplab":
        model = load_deeplab(args.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == "fair_cnn":
        model = load_fair_cnn(args.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.model_type == "fair_vit":
        model = load_fair_vit(args.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    #Debug to make sure we are using cuda
    print("Using device:", device)
    model.to(device)

    best_loss = float("inf")

    epoch = 0
    while epoch < args.epochs:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.model_type,
            args.num_classes,
        )

        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {train_loss}")

        f = open(log_path, "a")
        f.write(f"{epoch+1},{train_loss}\n")
        f.close()

        if train_loss < best_loss:
            best_loss = train_loss
            ckpt = os.path.join(args.save_path, f"{args.model_type}_best.pt")
            torch.save(model.state_dict(), ckpt)

        epoch += 1


if __name__ == "__main__":
    main()
