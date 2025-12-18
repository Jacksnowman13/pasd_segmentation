import os
import shutil
import random

VOC_ROOT = r"C:\projects\pasd_segmentation\data_raw\archive\VOC2012"

JPEG_DIR = os.path.join(VOC_ROOT, "JPEGImages")
SEG_DIR = os.path.join(VOC_ROOT, "SegmentationClass")

assert os.path.isdir(JPEG_DIR), f"No JPEGImages folder at {JPEG_DIR}"
assert os.path.isdir(SEG_DIR), f"No SegmentationClass folder at {SEG_DIR}"

PROJECT_ROOT = r"C:\projects\pasd_segmentation"
OUT_DATA = os.path.join(PROJECT_ROOT, "data")
OUT_TRAIN_IMG = os.path.join(OUT_DATA, "images_train")
OUT_TRAIN_MASK = os.path.join(OUT_DATA, "masks_train")
OUT_VAL_IMG = os.path.join(OUT_DATA, "images_val")
OUT_VAL_MASK = os.path.join(OUT_DATA, "masks_val")

os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
os.makedirs(OUT_TRAIN_MASK, exist_ok=True)
os.makedirs(OUT_VAL_IMG, exist_ok=True)
os.makedirs(OUT_VAL_MASK, exist_ok=True)

def copy_ids(id_list, out_img_dir, out_mask_dir):
    copied = 0
    for img_id in id_list:
        img_src = os.path.join(JPEG_DIR, img_id + ".jpg")
        mask_src = os.path.join(SEG_DIR, img_id + ".png")
        if not (os.path.exists(img_src) and os.path.exists(mask_src)):
            continue
        img_dst = os.path.join(out_img_dir, img_id + ".jpg")
        mask_dst = os.path.join(out_mask_dir, img_id + ".png")
        shutil.copy2(img_src, img_dst)
        shutil.copy2(mask_src, mask_dst)
        copied += 1

def main():
    mask_files = [
        f for f in os.listdir(SEG_DIR)
        if f.lower().endswith(".png")
    ]
    ids = [os.path.splitext(f)[0] for f in mask_files]

    random.seed(42)
    random.shuffle(ids)

    split_idx = int(0.8 * len(ids))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    copy_ids(train_ids, OUT_TRAIN_IMG, OUT_TRAIN_MASK)
    copy_ids(val_ids, OUT_VAL_IMG, OUT_VAL_MASK)

if __name__ == "__main__":
    main()
