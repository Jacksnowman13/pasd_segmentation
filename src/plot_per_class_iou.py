import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from metrics import compute_iou_from_confusion

VOC_CLASS_NAMES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
]

def load_iou(confmat_path):
    hist = torch.load(confmat_path, map_location="cpu")
    iou, miou = compute_iou_from_confusion(hist)
    return iou.numpy(), miou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confmat_a", type=str, required=True, help="Path to model A confmat .pt")
    parser.add_argument("--label_a", type=str, default="ModelA")
    parser.add_argument("--confmat_b", type=str, required=True, help="Path to model B confmat .pt")
    parser.add_argument("--label_b", type=str, default="ModelB")
    parser.add_argument("--out_name", type=str, default="per_class_iou_comparison.png")
    args = parser.parse_args()

    iou_a, miou_a = load_iou(args.confmat_a)
    iou_b, miou_b = load_iou(args.confmat_b)

    x = np.arange(len(VOC_CLASS_NAMES))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, iou_a, width, label=f"{args.label_a} (mIoU={miou_a:.3f})")
    plt.bar(x + width/2, iou_b, width, label=f"{args.label_b} (mIoU={miou_b:.3f})")
    plt.xticks(x, VOC_CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("IoU")
    plt.title("Per-class IoU comparison (VOC 2012)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", args.out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
