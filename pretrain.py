"""
Self-supervised pretraining of the pix2point backbone on the AASCE training
images (no labels; frozen 424 / 47 train / val split).

    python pretrain.py --data-root <AASCE_DATA> --output-dir runs/pretraining

Writes ``<output-dir>/train/``: ``unet_backbone_last.pth`` (last epoch: the
backbone given to ``train.py --pretrained-backbone``), ``unet_backbone_best.pth``
(lowest validation loss), ``config.json`` and ``pretrain_losses.png``.
"""

import argparse
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pix2point.config import PRETRAIN_CONFIG, SCHEMA_PATH, TRAIN_SPLIT_FILE, VAL_SPLIT_FILE  # noqa: E402
from pix2point.dataset import get_train_val_loaders, load_schema, read_split_file  # noqa: E402
from pix2point.model import Pix2Point  # noqa: E402
from train import set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="AASCE data root, containing train/{images_prepro_clahe,labels}")
    ap.add_argument("--output-dir", required=True, help="run directory (created)")
    ap.add_argument("--seed", type=int, default=PRETRAIN_CONFIG["seed"])
    ap.add_argument("--num-workers", type=int, default=PRETRAIN_CONFIG["num_workers"])
    args = ap.parse_args()

    print("Script run at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    cfg = copy.deepcopy(PRETRAIN_CONFIG)
    cfg["seed"] = args.seed
    cfg["num_workers"] = args.num_workers
    schema = load_schema(SCHEMA_PATH)
    cfg["n_classes"] = len(schema)   # sizes the (unused) prototype head, as in the original runs
    train_dir = os.path.join(os.path.abspath(args.output_dir), "train")
    os.makedirs(train_dir, exist_ok=True)
    cfg["train_dir"] = train_dir
    with open(os.path.join(train_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({**cfg, "data_root": os.path.abspath(args.data_root)}, fh, indent=4)
    cfg["schema"] = schema
    print(f"Run directory: {train_dir}")

    set_seed(cfg["seed"])
    print(f"Seed: {cfg['seed']} | batch_size: {cfg['batch_size']}")

    training_loader, validation_loader = get_train_val_loaders(
        args.data_root, schema, cfg, read_split_file(TRAIN_SPLIT_FILE), read_split_file(VAL_SPLIT_FILE),
        mode="pretrain")
    print(f"Images in train split: {len(training_loader.dataset)}")
    print(f"Images in val split:   {len(validation_loader.dataset)}")

    model = Pix2Point(cfg)
    print("Starting pretraining...")
    avg_train_losses, avg_val_losses = model.run_pretraining(training_loader, validation_loader)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_losses, label="Training Loss")
    plt.plot(avg_val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Barlow Twins loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(train_dir, "pretrain_losses.png"))
    plt.close()

    print(f"Backbone for train.py --pretrained-backbone: {os.path.join(train_dir, 'unet_backbone_last.pth')}")
    print("End at:", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
