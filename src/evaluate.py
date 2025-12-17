import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PASDDataset
from models_segformer import load_segformer
from models_deeplab import load_deeplab
from metrics import compute_confusion_matrix, compute_iou_from_confusion, compute_pixel_accuracy


def evaluate(model, val_loader, device, num_classes, model_type):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            if model_type == "segformer":
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                outputs = model(imgs)
                logits = outputs["out"]

            preds = torch.argmax(logits, dim=1)

            preds_list.extend(preds.cpu())
            labels_list.extend(masks.cpu())

    hist = compute_confusion_matrix(preds_list, labels_list, num_classes)
    iou_per_class, miou = compute_iou_from_confusion(hist)
    pixel_acc = compute_pixel_accuracy(hist)

    return iou_per_class, miou, pixel_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_img_dir", type=str, default=r"..\data_raw\val\img")
    parser.add_argument("--val_mask_dir", type=str, default=r"..\data_raw\val\ann")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--model_type", type=str, choices=["segformer", "deeplab"], default="segformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    val_ds = PASDDataset(args.val_img_dir, args.val_mask_dir, transform=None)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    if args.model_type == "segformer":
        model = load_segformer(args.num_classes)
    else:
        model = load_deeplab(args.num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    iou_per_class, miou, pixel_acc = evaluate(
        model, val_loader, device, args.num_classes, args.model_type
    )

    print(f"Mean IoU: {miou:.4f}")
    print(f"Pixel accuracy: {pixel_acc:.4f}")
    print("Per-class IoU:", iou_per_class.tolist())


if __name__ == "__main__":
    main()
