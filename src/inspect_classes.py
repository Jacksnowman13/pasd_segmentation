import os
import cv2
import numpy as np

ANN_DIR = r"C:\projects\pasd_segmentation\data_raw\train\ann"

values = set()

for i, fname in enumerate(os.listdir(ANN_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(ANN_DIR, fname)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    vals = np.unique(mask)
    values.update(vals.tolist())

    if i >= 50:
        break

print("Unique label values in first ~50 masks:", sorted(values))
print("Number of distinct labels:", len(values))
