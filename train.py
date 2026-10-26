"""
Train pix2point on AASCE (frozen 424 / 47 train / val split of the training set).

From scratch::

    python train.py --data-root <AASCE_DATA> --output-dir runs/scratch

From a pretrained backbone (only the backbone initialisation differs)::

    python train.py --data-root <AASCE_DATA> --output-dir runs/pretrained \
        --pretrained-backbone weights/unet_backbone_last.pth

Writes ``<output-dir>/train/``: ``config.json``, the three checkpoints
(``unet_backbone[_best_mre|_last].pth`` + ``prototype_layer[_best_mre|_last].pth``),
``heatmaps_final/`` and the loss / distance plots.
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pix2point.config import CONFIG, SCHEMA_PATH, TRAIN_SPLIT_FILE, VAL_SPLIT_FILE  # noqa: E402
from pix2point.dataset import get_train_val_loaders, load_schema, read_split_file  # noqa: E402
from pix2point.model import Pix2Point  # noqa: E402


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training (weight init, batch shuffling,
    augmentations: the albumentations pipelines are seeded from NumPy's RNG,
    see ``SpineDataset``)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="AASCE data root, containing train/{images_prepro_clahe,labels}")
    ap.add_argument("--output-dir", required=True, help="run directory (created)")
    ap.add_argument("--pretrained-backbone", default=None,
                    help="backbone weights to start from (omit to train from scratch)")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    args = ap.parse_args()

    print("Script run at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.pretrained_backbone and not os.path.isfile(args.pretrained_backbone):
        raise FileNotFoundError(f"Pretrained backbone not found: {args.pretrained_backbone}")

    cfg = copy.deepcopy(CONFIG)
    cfg["seed"] = args.seed
    cfg["num_workers"] = args.num_workers
    schema = load_schema(SCHEMA_PATH)
    cfg["n_classes"] = len(schema)
    cfg["weights_backbone_load_path"] = os.path.abspath(args.pretrained_backbone) if args.pretrained_backbone else None
    train_dir = os.path.join(os.path.abspath(args.output_dir), "train")
    os.makedirs(train_dir, exist_ok=True)
    cfg["train_dir"] = train_dir

    # Snapshot of the configuration used for this run.
    snapshot = {**cfg, "data_root": os.path.abspath(args.data_root),
                "backbone_init": "pretrained" if args.pretrained_backbone else "scratch"}
    with open(os.path.join(train_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=4)
    cfg["schema"] = schema

    print(f"Run directory: {train_dir}")
    print(f"Backbone init: {snapshot['backbone_init']}"
          + (f" <- {cfg['weights_backbone_load_path']}" if args.pretrained_backbone else ""))

    set_seed(cfg["seed"])
    print(f"Seed: {cfg['seed']}")
    print(f"Landmark schema: {len(schema)} classes | "
          f"target_size=({cfg['target_height']}, {cfg['target_width']}) | batch_size: {cfg['batch_size']}")

    train_subjects = read_split_file(TRAIN_SPLIT_FILE)
    val_subjects = read_split_file(VAL_SPLIT_FILE)
    training_loader, validation_loader = get_train_val_loaders(
        args.data_root, schema, cfg, train_subjects, val_subjects)
    print(f"Images in train split: {len(training_loader.dataset)}")
    print(f"Images in val split:   {len(validation_loader.dataset)}")

    model = Pix2Point(cfg)
    print("Starting training...")
    avg_train_losses, avg_val_losses = model.run_finetuning(training_loader, validation_loader)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_losses, label="Training Loss")
    plt.plot(avg_val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(train_dir, "losses.png"))
    plt.close()

    print("End at:", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
