from collections import Counter
import torch
import albumentations as A
from dataset import PASDDataset

transform = A.Compose([A.Resize(512, 512)])

val_ds = PASDDataset(r"..\data\images_val", r"..\data\masks_val", transform=transform)

label_counts = Counter()

for _, mask in val_ds:
    unique, counts = torch.unique(mask, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        label_counts[u] += c

print(label_counts)
