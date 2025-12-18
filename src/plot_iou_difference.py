import os
import sys
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


# AI assisted
def main():
    confmat_ref = sys.argv[1]
    label_ref = sys.argv[2]
    confmat_cmp = sys.argv[3]
    label_cmp = sys.argv[4]
    out_name = sys.argv[5]

    iou_ref, miou_ref = load_iou(confmat_ref)
    iou_cmp, miou_cmp = load_iou(confmat_cmp)

    diff = iou_cmp - iou_ref
    x = np.arange(len(VOC_CLASS_NAMES))

    plt.figure(figsize=(12, 4))
    plt.axhline(0.0, linewidth=1)
    plt.bar(x, diff)
    plt.xticks(x, VOC_CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel(f"ΔIoU ({label_cmp} − {label_ref})")
    plt.title(f"Per-class IoU difference (mIoU: {label_cmp}={miou_cmp}, {label_ref}={miou_ref})")
    plt.tight_layout()

    os.makedirs(r"..\figures", exist_ok=True)
    out_path = os.path.join(r"..\figures", out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
