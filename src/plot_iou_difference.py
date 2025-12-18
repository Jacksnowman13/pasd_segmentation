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
    # Argument parsing AI assisted
    parser = argparse.ArgumentParser()
    parser.add_argument("--confmat_ref", type=str, required=True, help="Reference model confmat .pt (subtracted)")
    parser.add_argument("--label_ref", type=str, default="Ref")
    parser.add_argument("--confmat_cmp", type=str, required=True, help="Comparison model confmat .pt (minus ref)")
    parser.add_argument("--label_cmp", type=str, default="Cmp")
    parser.add_argument("--out_name", type=str, default="delta_iou.png")
    args = parser.parse_args()

    iou_ref, miou_ref = load_iou(args.confmat_ref)
    iou_cmp, miou_cmp = load_iou(args.confmat_cmp)

    diff = iou_cmp - iou_ref
    x = np.arange(len(VOC_CLASS_NAMES))

    plt.figure(figsize=(12, 4))
    plt.axhline(0.0, linewidth=1)
    plt.bar(x, diff)
    plt.xticks(x, VOC_CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel(f"ΔIoU ({args.label_cmp} − {args.label_ref})")
    plt.title(f"Per-class IoU difference (mIoU: {args.label_cmp}={miou_cmp}, {args.label_ref}={miou_ref})")
    plt.tight_layout()

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", args.out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
