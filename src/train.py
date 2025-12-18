import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

from dataset import PASDDataset
from models_fair import load_fair_cnn, load_fair_vit


def get_train_loader(train_img_dir, train_mask_dir, batch_size):
    tfm = A.Compose([A.Resize(512, 512)])
    ds = PASDDataset(train_img_dir, train_mask_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def train_one_epoch(model, loader, opt, device, num_classes):
    model.train()
    total = 0.0
    for imgs, masks in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        opt.zero_grad()

        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        targets = masks.clone().long()
        bad = (targets < 0) | (targets >= num_classes)
        targets[bad] = 255

        loss = F.cross_entropy(logits, targets, ignore_index=255)
        loss.backward()
        opt.step()

        total += loss.item()
    return total / max(1, len(loader))


# AI assisted
def main():
    model_type = sys.argv[1]
    num_classes = int(sys.argv[2])
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    lr = float(sys.argv[5])
    train_img_dir = sys.argv[6]
    train_mask_dir = sys.argv[7]
    save_path = sys.argv[8]

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(r"..\logs", exist_ok=True)

    log_path = os.path.join(r"..\logs", f"{model_type}_loss.csv")
    f = open(log_path, "w")
    f.write("epoch,loss\n")
    f.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Debug to see what device
    print("Using device:", device)

    loader = get_train_loader(train_img_dir, train_mask_dir, batch_size)

    if model_type == "fair_cnn":
        model = load_fair_cnn(num_classes)
    else:
        model = load_fair_vit(num_classes)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best = 1e30

    for ep in range(epochs):
        loss = train_one_epoch(model, loader, opt, device, num_classes)
        print("Epoch", ep + 1, "/", epochs, "- train loss:", loss)

        f = open(log_path, "a")
        f.write(str(ep + 1) + "," + str(loss) + "\n")
        f.close()

        if loss < best:
            best = loss
            ckpt = os.path.join(save_path, f"{model_type}_best.pt")
            torch.save(model.state_dict(), ckpt)


if __name__ == "__main__":
    main()
