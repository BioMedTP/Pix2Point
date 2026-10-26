"""
AASCE dataset and dataloaders.

Expected layout of the data root (see README)::

    <data_root>/<subset>/images_prepro_clahe/<subject>.nii      752 x 256, uint8 range
    <data_root>/<subset>/labels/<subject>.json                  {vertebra: {corner: [x, y]}}
    <data_root>/<subset>/cropping_info/<subject>_crop_info.json (validation MRE and evaluation)

with ``<subset>`` in ``train`` / ``test``. Label coordinates are pixels of the
752 x 256 image. Targets are flat spatial indices ``y * W + x``, absent or
out-of-frame landmarks being :data:`IGNORE_INDEX`. In ``pretrain`` mode the
labels only pair the subjects: a sample is two augmented views of the image.
"""

from __future__ import annotations

import copy
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import nibabel as nib
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Subset, get_worker_info

IGNORE_INDEX = -100


def load_schema(path: str) -> List[Tuple[str, str]]:
    """Ordered ``(vertebra, corner)`` landmark schema."""
    with open(path, encoding="utf-8") as fh:
        return [tuple(pair) for pair in json.load(fh)]


def read_split_file(path: str) -> List[str]:
    """Read a subject list: one subject per line, '#' starts a comment line."""
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def subject_key(path: str) -> str:
    """Subject key of an image or label file: its basename without extension."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii", ".json"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def load_nifti_uint8(path: str) -> np.ndarray:
    """Read a preprocessed 2-D NIfTI image as ``uint8`` (no re-windowing)."""
    arr = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D NIfTI image, got shape {arr.shape}: {path}")
    return np.clip(arr, 0, 255).astype(np.uint8)


def parse_landmarks(json_path: str, schema_index: Dict[Tuple[str, str], int]
                    ) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Flatten a ``{vertebra: {corner: [x, y]}}`` label into keypoints + class indices."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    keypoints: List[Tuple[float, float]] = []
    class_labels: List[int] = []
    for group, points in data.items():
        for point_name, (x, y) in points.items():
            cls = schema_index.get((group, point_name))
            if cls is None:               # point not in schema -> ignore
                continue
            keypoints.append((float(x), float(y)))
            class_labels.append(cls)
    return keypoints, class_labels


class SpineDataset(Dataset):
    """Images + landmark targets of one subset, sorted by subject key.

    ``transform_params`` (the protocol config) enables the training
    augmentations; ``None`` disables them (validation / test). ``mode``
    ``"finetune"`` returns ``(image, targets)``; ``"pretrain"`` returns two
    views ``(view1, view2)``, augmented independently (intensity only).
    """

    def __init__(
        self,
        root_dir: str,
        schema: List[Tuple[str, str]],
        target_size: Tuple[int, int],
        subset: str,
        images_dirname: str,
        labels_dirname: str,
        transform_params: Optional[dict] = None,
        mode: str = "finetune",
    ) -> None:
        if mode not in ("finetune", "pretrain"):
            raise ValueError(f"mode must be 'finetune' or 'pretrain', got {mode!r}")
        self.mode = mode
        self.img_dir = os.path.join(root_dir, subset, images_dirname)
        self.labels_dir = os.path.join(root_dir, subset, labels_dirname)
        self.target_h, self.target_w = target_size
        self.schema_index = {pair: idx for idx, pair in enumerate(schema)}
        self.n_points = len(schema)

        self.samples = self._pair_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No image/label pairs found under {self.img_dir} / {self.labels_dir}"
            )
        self.transform = self._build_transform(transform_params)
        # albumentations' Compose has its own RNG, ignored by the global seeds:
        # seed it from NumPy's global RNG (seeded by set_seed) for reproducible runs.
        self.transform.set_random_seed(int(np.random.randint(0, 2**31 - 1)))
        self._aug_worker_seed = None

    def _pair_samples(self) -> List[Tuple[str, str]]:
        """``(image_path, label_path)`` pairs, sorted by subject key."""
        images = (glob.glob(os.path.join(self.img_dir, "*.nii"))
                  + glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        labels = glob.glob(os.path.join(self.labels_dir, "*.json"))
        img_by_key = {subject_key(p): p for p in images}
        lbl_by_key = {subject_key(p): p for p in labels}
        common = sorted(set(img_by_key) & set(lbl_by_key))
        return [(img_by_key[k], lbl_by_key[k]) for k in common]

    def _build_transform(self, params: Optional[dict]) -> A.Compose:
        """[geometric] -> pad/crop to the canonical size -> [intensity] -> normalize -> tensor."""
        augment = params is not None
        steps: List[A.BasicTransform] = []
        if augment and self.mode == "finetune":
            steps.extend(self._geometric_augmentations(params))
        # Zero-pad then center-crop: a strict no-op for 752 x 256 images.
        steps.extend([
            A.PadIfNeeded(
                min_height=self.target_h, min_width=self.target_w,
                border_mode=cv2.BORDER_CONSTANT, position="center", fill=0, p=1.0,
            ),
            A.CenterCrop(height=self.target_h, width=self.target_w, p=1.0),
        ])
        if augment:
            gamma_limit = float(params["random_gamma_limit"])
            steps.extend([
                A.GaussianBlur(blur_limit=0,
                               sigma_limit=(0.1, max(0.1, float(params["random_blur_std"]))),
                               p=float(params["random_blur_p"])),
                A.GaussNoise(std_range=(0.01, max(0.02, float(params["random_noise_std"]))),
                             p=float(params["random_noise_p"])),
                A.RandomGamma(gamma_limit=(100.0 - gamma_limit, 100.0 + gamma_limit),
                              p=float(params["random_gamma_p"])),
                A.RandomBrightnessContrast(brightness_limit=float(params["random_brightness_limit"]),
                                           contrast_limit=float(params["random_contrast_limit"]),
                                           p=float(params["random_brightness_contrast_p"])),
            ])
        steps.extend([
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        if self.mode == "pretrain":
            return A.Compose(steps)
        return A.Compose(
            steps,
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False, label_fields=["class_labels"]
            ),
        )

    def _geometric_augmentations(self, params: dict) -> List[A.BasicTransform]:
        """Random zoom-in (random crop or affine scale) and rotation, see ``CONFIG``."""
        rot_deg = float(params["max_rotation_deg"])
        zoom = max(0.0, float(params["max_zoom"]))
        shift = max(0.0, float(params["max_translation"]))
        p = float(params["geometric_p"])
        if params["random_crop"]:
            ratio = float(self.target_w) / float(self.target_h)
            return [
                # Random patch of 1 / (1 + zoom) to 100 % of the area, resized
                # back to the canonical size; the fixed ratio prevents stretching.
                A.RandomResizedCrop(size=(self.target_h, self.target_w),
                                    scale=(1.0 / (1.0 + zoom), 1.0), ratio=(ratio, ratio), p=p),
                A.Affine(rotate=(-rot_deg, rot_deg), border_mode=cv2.BORDER_CONSTANT, fill=0, p=p),
            ]
        return [A.Affine(
            scale=(1.0, 1.0 + zoom),          # zoom-in only
            translate_percent={"x": (-shift, shift), "y": (-shift, shift)},
            rotate=(-rot_deg, rot_deg),
            border_mode=cv2.BORDER_CONSTANT, fill=0, p=p,
        )]

    def __len__(self) -> int:
        return len(self.samples)

    def _seed_transform_in_worker(self) -> None:
        """Re-seed the augmentation RNG once per DataLoader worker.

        Workers are forked each epoch with a copy of the transform whose RNG is
        in the same state: without this, every worker of every epoch would
        replay the same augmentations. ``worker_info.seed`` is derived by
        PyTorch from the global torch seed and differs per worker and epoch.
        """
        info = get_worker_info()
        if info is not None and self._aug_worker_seed != info.seed:
            self._aug_worker_seed = info.seed
            self.transform.set_random_seed(info.seed % 2**32)

    def _generate_targets(self, keypoints: np.ndarray, labels: List[int]) -> torch.Tensor:
        """``(n_points,)`` flat indices ``y * W + x`` of the rounded keypoints."""
        targets = np.full((self.n_points,), IGNORE_INDEX, dtype=np.int64)
        for (x, y), cls in zip(keypoints, labels):
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < self.target_w and 0 <= yi < self.target_h:
                targets[int(cls)] = yi * self.target_w + xi
        return torch.from_numpy(targets)

    def __getitem__(self, idx: int):
        self._seed_transform_in_worker()
        img_path, lbl_path = self.samples[idx]
        image = load_nifti_uint8(img_path)
        if self.mode == "pretrain":
            view1 = self.transform(image=image.copy())["image"]
            view2 = self.transform(image=image.copy())["image"]
            return ((view1.unsqueeze(0) if view1.ndim == 2 else view1),
                    (view2.unsqueeze(0) if view2.ndim == 2 else view2))
        keypoints, class_labels = parse_landmarks(lbl_path, self.schema_index)
        transformed = self.transform(image=image, keypoints=keypoints, class_labels=class_labels)
        img_tensor = transformed["image"]
        if img_tensor.ndim == 2:                      # ensure channel dim (1, H, W)
            img_tensor = img_tensor.unsqueeze(0)
        kpts = np.asarray(transformed["keypoints"], dtype=np.float32).reshape(-1, 2)
        return img_tensor, self._generate_targets(kpts, transformed["class_labels"])


def _subject_indices(samples: List[Tuple[str, str]], subjects: List[str]) -> List[int]:
    """Indices of ``subjects`` in ``samples``, in the order of ``subjects``."""
    index_by_key = {subject_key(img): i for i, (img, _) in enumerate(samples)}
    missing = [s for s in subjects if s not in index_by_key]
    if missing:
        raise ValueError(f"{len(missing)} split subject(s) not found in the dataset: {missing[:5]}")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Duplicate subject(s) in the split list.")
    return [index_by_key[s] for s in subjects]


def get_train_val_loaders(root_dir: str, schema: List[Tuple[str, str]], cfg: dict,
                          train_subjects: List[str], val_subjects: List[str],
                          mode: str = "finetune") -> Tuple[DataLoader, DataLoader]:
    """Training (augmented, shuffled) and validation loaders, both from ``<root>/train``."""
    if set(train_subjects) & set(val_subjects):
        raise ValueError("train_subjects and val_subjects overlap.")
    target_size = (cfg["target_height"], cfg["target_width"])
    train_ds = SpineDataset(root_dir, schema, target_size, "train",
                            cfg["images_dirname"], cfg["labels_dirname"], cfg, mode)
    val_ds = SpineDataset(root_dir, schema, target_size, "train",
                          cfg["images_dirname"], cfg["labels_dirname"], None, mode)
    train_idx = _subject_indices(train_ds.samples, train_subjects)
    val_idx = _subject_indices(val_ds.samples, val_subjects)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])
    return train_loader, val_loader


def non_augmented_loader(training_loader: DataLoader) -> DataLoader:
    """Same samples as ``training_loader``, not augmented and not shuffled."""
    subset = training_loader.dataset
    clean = copy.copy(subset.dataset)
    clean.transform = clean._build_transform(None)
    return DataLoader(Subset(clean, subset.indices), batch_size=training_loader.batch_size,
                      shuffle=False, num_workers=training_loader.num_workers)


def inverse_resize_scales(loader: DataLoader) -> torch.Tensor:
    """``1 / resize_scale`` of each sample of a non-shuffled loader, in its order.

    Read from ``<subset>/cropping_info/<subject>_crop_info.json``. The mapping
    back to the original radiograph is a uniform scaling plus a shift, so a
    distance in the original image is the 752 x 256 distance times this value.
    """
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("inverse_resize_scales needs a loader that is not shuffled")
    subset = loader.dataset
    scales = []
    for i in subset.indices:
        label_path = subset.dataset.samples[i][1]
        crop_path = os.path.join(os.path.dirname(os.path.dirname(label_path)), "cropping_info",
                                 f"{subject_key(label_path)}_crop_info.json")
        with open(crop_path, encoding="utf-8") as fh:
            scales.append(1.0 / float(json.load(fh)["resize_scale"]))
    return torch.tensor(scales, dtype=torch.float64)
