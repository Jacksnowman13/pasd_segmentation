import torch

def _fast_hist(pred, label, num_classes):
    # Ignore labels outside [0, num_classes-1] (e.g., 255 = ignore)
    mask = (label >= 0) & (label < num_classes)
    hist = torch.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist

def compute_confusion_matrix(preds, labels, num_classes):
    hist = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, l in zip(preds, labels):
        hist += _fast_hist(p.view(-1), l.view(-1), num_classes)
    return hist

def compute_iou_from_confusion(hist):
    """
    Returns:
      iou: per-class IoU, length = num_classes
      miou: mean IoU over classes that actually appear (union > 0)
    """
    hist = hist.float()
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp

    denom = tp + fp + fn  # no eps here
    valid = denom > 0      # classes that appear in GT or preds

    iou = torch.zeros_like(tp)
    iou[valid] = tp[valid] / denom[valid]

    if valid.any():
        miou = iou[valid].mean().item()
    else:
        miou = 0.0

    return iou, miou

def compute_pixel_accuracy(hist):
    hist = hist.float()
    correct = torch.diag(hist).sum()
    total = hist.sum() + 1e-6
    return (correct / total).item()
