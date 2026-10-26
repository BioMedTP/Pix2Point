#!/usr/bin/env python3
"""Evaluate AASCE test predictions with the metrics of the paper.

Input: a CSV ``image_id,landmark_id,x,y``, coordinates in the 752 x 256 input
image, ``landmark_id`` in [0, 67] in the order of the landmark schema.

Predictions AND ground truth are reprojected to the original radiograph with
each subject's ``cropping_info``:

    x_orig = (x - pad_left) / resize_scale + left
    y_orig = (y - pad_top)  / resize_scale + top

AASCE gives no pixel spacing, so each radial error is divided by the length
of the subject's spine (polyline from the midpoint of the T1 upper endplate,
through the vertebral centres (mean of the 4 corners), to the midpoint of the
L5 lower endplate, on the ground truth). Metrics:
  - MRE_norm (%): mean of these normalised errors, +/- standard deviation over
    all landmarks of the test set;
  - SDR@1% (%): percentage of landmarks whose error is at most 1 % of the
    spine length (denominator: all expected landmarks, a missing one counts
    as a failure).

Usage::

    python evaluate.py --data-root <AASCE_DATA> --pred <run>/test/aasce_test.csv \
        --out <run>/test/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pix2point.config import CONFIG, SCHEMA_PATH  # noqa: E402
from pix2point.dataset import load_schema, subject_key  # noqa: E402

SDR_SPINE_PCT = 1.0     # SDR threshold, in % of the subject's spine length
PRED_COLUMNS = ("image_id", "landmark_id", "x", "y")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def list_subjects(data_root: str, split: str = "test") -> List[str]:
    """Subjects of a split, sorted, present in images, labels and cropping_info."""
    base = os.path.join(data_root, split)
    images = {subject_key(f) for f in os.listdir(os.path.join(base, CONFIG["images_dirname"]))
              if f.endswith((".nii", ".nii.gz"))}
    labels = {f[: -len(".json")] for f in os.listdir(os.path.join(base, CONFIG["labels_dirname"]))
              if f.endswith(".json")}
    crops = {f[: -len("_crop_info.json")] for f in os.listdir(os.path.join(base, "cropping_info"))
             if f.endswith("_crop_info.json")}
    return sorted(images & labels & crops)


def load_landmarks(data_root: str, subject: str, split: str, schema) -> np.ndarray:
    """``(68, 2)`` ground truth (x, y) in the 752 x 256 image, in schema order."""
    path = os.path.join(data_root, split, CONFIG["labels_dirname"], f"{subject}.json")
    with open(path) as fh:
        raw = json.load(fh)
    pts = np.empty((len(schema), 2), dtype=np.float64)
    for i, (vertebra, corner) in enumerate(schema):
        xy = raw[vertebra][corner]
        pts[i] = (float(xy[0]), float(xy[1]))
    return pts


def load_crop_info(data_root: str, subject: str, split: str) -> Dict:
    path = os.path.join(data_root, split, "cropping_info", f"{subject}_crop_info.json")
    with open(path) as fh:
        return json.load(fh)


def to_original(points: np.ndarray, crop_info: Dict) -> np.ndarray:
    """752 x 256 -> original image."""
    pts = np.asarray(points, dtype=np.float64)
    scale = float(crop_info["resize_scale"])
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - float(crop_info["pad_left"])) / scale + float(crop_info["left"])
    out[:, 1] = (pts[:, 1] - float(crop_info["pad_top"])) / scale + float(crop_info["top"])
    return out


def load_predictions_csv(path: str, n_landmarks: int) -> Dict[str, np.ndarray]:
    """``{image_id: (n_landmarks, 2)}``; absent landmarks are NaN."""
    preds: Dict[str, np.ndarray] = {}
    seen: Dict[str, set] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in PRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row_no, row in enumerate(reader, start=2):
            image_id = row["image_id"].strip()
            lid = int(row["landmark_id"])
            if not 0 <= lid < n_landmarks:
                raise ValueError(f"{path}:{row_no}: landmark_id={lid} out of [0, {n_landmarks - 1}]")
            if image_id not in preds:
                preds[image_id] = np.full((n_landmarks, 2), np.nan, dtype=np.float64)
                seen[image_id] = set()
            if lid in seen[image_id]:
                raise ValueError(f"{path}:{row_no}: duplicated landmark_id={lid} for {image_id}")
            seen[image_id].add(lid)
            preds[image_id][lid] = (float(row["x"]), float(row["y"]))
    return preds


def write_predictions_csv(path: str, predictions: Dict[str, np.ndarray]) -> None:
    """Write ``image_id,landmark_id,x,y``, sorted by image_id; NaN points are omitted.

    Coordinates are written with ``repr`` (exact float64 round trip): a
    truncation would be amplified by up to 1 / resize_scale at reprojection.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(PRED_COLUMNS)
        for image_id in sorted(predictions):
            pts = np.asarray(predictions[image_id], dtype=np.float64)
            for lid in range(pts.shape[0]):
                x, y = pts[lid]
                if np.isnan(x) or np.isnan(y):
                    continue
                writer.writerow([image_id, lid, repr(float(x)), repr(float(y))])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def spine_length(points_orig: np.ndarray, schema) -> float:
    """Spine length (unit of ``points_orig``): T1 upper-endplate midpoint ->
    vertebral centres, top to bottom -> L5 lower-endplate midpoint."""
    vertebrae: List[str] = []
    for group, _ in schema:
        if group not in vertebrae:
            vertebrae.append(group)
    idx = {v: [i for i, (g, _) in enumerate(schema) if g == v] for v in vertebrae}
    pts = np.asarray(points_orig, dtype=np.float64)

    def corners(v, prefix):
        return [i for i in idx[v] if schema[i][1].startswith(prefix)]

    path = [pts[corners(vertebrae[0], "Plat_Sup")].mean(axis=0)]
    path += [pts[idx[v]].mean(axis=0) for v in vertebrae]
    path.append(pts[corners(vertebrae[-1], "Plat_Inf")].mean(axis=0))
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def evaluate(data_root: str, pred_path: str, split: str = "test") -> Dict:
    """MRE_norm (%) +/- std and SDR@1% (%) of a prediction CSV."""
    schema = load_schema(SCHEMA_PATH)
    n_lm = len(schema)
    preds = load_predictions_csv(pred_path, n_lm)
    expected = list_subjects(data_root, split)

    extra_subjects = sorted(set(preds) - set(expected))
    missing_subjects = sorted(set(expected) - set(preds))
    if extra_subjects:
        raise ValueError(f"{pred_path}: {len(extra_subjects)} subject(s) outside the {split} split")
    if missing_subjects:
        raise ValueError(f"{pred_path}: {len(missing_subjects)} subject(s) without prediction: "
                         f"{missing_subjects[:5]}")

    err_pct = np.full((len(expected), n_lm), np.nan)    # error in % of the spine length; NaN = missing
    for i, subject in enumerate(expected):
        p_in = preds[subject]
        crop = load_crop_info(data_root, subject, split)
        g_or = to_original(load_landmarks(data_root, subject, split, schema), crop)
        p_or = to_original(np.nan_to_num(p_in, nan=0.0), crop)
        p_or[np.isnan(p_in).any(axis=1)] = np.nan
        err_pct[i] = np.sqrt(((p_or - g_or) ** 2).sum(axis=-1)) / spine_length(g_or, schema) * 100.0

    valid = err_pct[~np.isnan(err_pct)]
    return {
        "predictions_file": os.path.abspath(pred_path),
        "n_subjects": len(expected),
        "n_landmarks_expected": int(err_pct.size),
        "n_landmarks_missing": int(err_pct.size - valid.size),
        "mre_norm_pct": float(valid.mean()),
        "mre_norm_pct_std": float(valid.std(ddof=0)),
        f"sdr_{SDR_SPINE_PCT:g}pct": float((valid <= SDR_SPINE_PCT).sum()) / err_pct.size * 100.0,
    }


def format_report(res: Dict) -> str:
    return "\n".join([
        f"Subjects evaluated : {res['n_subjects']}",
        f"Landmarks expected : {res['n_landmarks_expected']}  (missing: {res['n_landmarks_missing']})",
        f"MRE_norm           : {res['mre_norm_pct']:.2f} +/- {res['mre_norm_pct_std']:.2f} % of spine length",
        f"SDR@{SDR_SPINE_PCT:g}%             : {res[f'sdr_{SDR_SPINE_PCT:g}pct']:.1f} %",
    ])


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="AASCE data root, containing test/{images_prepro_clahe,labels,cropping_info}")
    ap.add_argument("--pred", required=True, help="CSV image_id,landmark_id,x,y")
    ap.add_argument("--out", default=None, help="output JSON")
    args = ap.parse_args(argv)

    res = evaluate(args.data_root, args.pred)
    print(format_report(res))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"Metrics written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
