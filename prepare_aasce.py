"""
Preprocess the official AASCE 2019 release into the layout read by the scripts
of this repository.

Input (official AASCE layout)::

    <AASCE_ROOT>/data/{training,test}/<name>.jpg
    <AASCE_ROOT>/labels/{training,test}/<name>.jpg.mat     key "p2": (68, 2) pixel coordinates

Output::

    <OUTPUT_DIR>/{train,test}/images_prepro_clahe/<subject>.nii
    <OUTPUT_DIR>/{train,test}/labels/<subject>.json
    <OUTPUT_DIR>/{train,test}/cropping_info/<subject>_crop_info.json

For each image: intensity normalisation (percentiles), resize to fit
752 x 256 (aspect ratio kept), CLAHE on the resized image, zero padding
centred to 752 x 256. The landmarks undergo the same resize and padding. No
cropping: the AASCE images are already cropped on the spine. The subjects
listed in data_invalid/ are skipped.

    python prepare_aasce.py --aasce-root <AASCE_ROOT> --output-dir <DATA_ROOT>
"""

import argparse
import json
import os
import re

import cv2
import nibabel as nib
import numpy as np
import scipy.io as sio

# ----- Constants ---------------------------------------------------------------
TARGET_H, TARGET_W = 752, 256              # network input (height, width)
PERCENTILES = (1, 99)                      # intensity window before the resize
NORM_EPS = 1e-8                            # avoids a division by zero on a flat image
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
# 17 vertebrae x 4 corners = the 68 points of "p2", grouped by vertebra.
VERTEBRAE = [f"Vertebre_T{i}" for i in range(1, 13)] + [f"Vertebre_L{i}" for i in range(1, 6)]
# Corners of a vertebra in the order of "p2": upper-left, upper-right, lower-left,
# lower-right of the image. The left of the image is the patient's right (D = droit,
# G = gauche), hence the names.
CORNER_NAMES = ["Plat_Sup_D", "Plat_Sup_G", "Plat_Inf_D", "Plat_Inf_G"]
AASCE_SUBSETS = {"train": "training", "test": "test"}
EXCLUDED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_invalid")
IMAGES_DIRNAME = "images_prepro_clahe"


def subject_id(image_file: str) -> str:
    """``sunhl-1th-02-Jan-2017-162 A AP.jpg`` -> ``sunhl-1th-02-Jan-2017-162_A_AP``."""
    base = os.path.splitext(os.path.basename(image_file))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")


def load_landmarks(mat_path: str) -> dict:
    """``{vertebra: {corner: [x, y]}}`` in pixels of the source image.

    The 17 groups of 4 points are sorted from top to bottom before being named:
    the order of the file is not reliable (on some images the last group is the
    highest vertebra).
    """
    pts = np.asarray(sio.loadmat(mat_path)["p2"], dtype=np.float64)   # p2 is uint16
    pts = pts.reshape(len(VERTEBRAE), len(CORNER_NAMES), 2)
    pts = pts[np.argsort(pts[:, :, 1].mean(axis=1), kind="stable")]
    return {v: {c: [float(pts[i, k, 0]), float(pts[i, k, 1])] for k, c in enumerate(CORNER_NAMES)}
            for i, v in enumerate(VERTEBRAE)}


def read_image(path: str) -> np.ndarray:
    """Grayscale image, windowed between the PERCENTILES and rescaled to uint8."""
    pixels = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    pixels = pixels.astype(np.float32)
    lo, hi = np.percentile(pixels, PERCENTILES)
    # Explicit float32 arithmetic: identical results under NumPy 1 and NumPy 2,
    # whose promotion rules (NEP 50) would otherwise compute this in float64.
    den = np.float32(hi - lo + NORM_EPS)
    lo, hi = np.float32(lo), np.float32(hi)
    return ((np.clip(pixels, lo, hi) - lo) / den * np.float32(255)).astype(np.uint8)


def preprocess(image: np.ndarray, landmarks: dict):
    """Resize (aspect ratio kept) + CLAHE + centred zero padding to TARGET_H x TARGET_W.

    Returns the image, the landmarks in the same frame, and the cropping info
    that maps this frame back to the source image.
    """
    h, w = image.shape
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    roi = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    pad_left, pad_top = (TARGET_W - new_w) // 2, (TARGET_H - new_h) // 2

    # CLAHE before padding, so that the black bands do not bias its histograms.
    roi = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID).apply(roi)
    out = np.pad(roi, ((pad_top, TARGET_H - new_h - pad_top), (pad_left, TARGET_W - new_w - pad_left)),
                 mode="constant", constant_values=0)

    labels = {v: {c: [x * scale + pad_left, y * scale + pad_top] for c, (x, y) in pts.items()}
              for v, pts in landmarks.items()}
    crop_info = {"resize_scale": float(scale), "pad_left": int(pad_left), "pad_top": int(pad_top),
                 "left": 0.0, "top": 0.0}
    return out, labels, crop_info


def excluded_subjects(split: str) -> set:
    with open(os.path.join(EXCLUDED_DIR, f"{split}.json"), encoding="utf-8") as fh:
        return {e["subject"] for e in json.load(fh)["excluded"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aasce-root", required=True, help="official AASCE release (data/, labels/)")
    ap.add_argument("--output-dir", required=True, help="DATA_ROOT given to the other scripts")
    args = ap.parse_args()

    for split, aasce_subset in AASCE_SUBSETS.items():
        img_dir = os.path.join(args.aasce_root, "data", aasce_subset)
        lbl_dir = os.path.join(args.aasce_root, "labels", aasce_subset)
        out = {d: os.path.join(args.output_dir, split, d) for d in (IMAGES_DIRNAME, "labels", "cropping_info")}
        for d in out.values():
            os.makedirs(d, exist_ok=True)
        excluded = excluded_subjects(split)

        n = 0
        for image_file in sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")):
            subject = subject_id(image_file)
            if subject in excluded:
                continue
            image = read_image(os.path.join(img_dir, image_file))
            landmarks = load_landmarks(os.path.join(lbl_dir, f"{image_file}.mat"))
            image, labels, crop_info = preprocess(image, landmarks)

            nib.save(nib.Nifti1Image(image, affine=np.eye(4)), os.path.join(out[IMAGES_DIRNAME], f"{subject}.nii"))
            with open(os.path.join(out["labels"], f"{subject}.json"), "w", encoding="utf-8") as fh:
                json.dump(labels, fh, indent=4)
            with open(os.path.join(out["cropping_info"], f"{subject}_crop_info.json"), "w", encoding="utf-8") as fh:
                json.dump(crop_info, fh, indent=4)
            n += 1
        print(f"{split}: {n} images written, {len(excluded)} excluded -> {os.path.join(args.output_dir, split)}")


if __name__ == "__main__":
    main()
