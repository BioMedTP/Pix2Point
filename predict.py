"""
Inference of a trained pix2point model on the 126 AASCE test images, then
evaluation (``evaluate.py``).

    python predict.py --data-root <AASCE_DATA> --weights-dir <dir> \
        --checkpoint best_mre --output-dir <out>

``--weights-dir`` holds ``unet_backbone<suffix>.pth`` and
``prototype_layer<suffix>.pth``, with suffix ``_best_mre`` (``--checkpoint
best_mre``, the checkpoint of the reported results), ``""`` (``best_loss``) or
``_last`` (``last``); a training run writes them in ``<run>/train/``, with the
``config.json`` from which the prototype head settings are read (``CONFIG``
otherwise).

Each landmark is predicted at the argmax of its heatmap (integer pixel of the
752 x 256 input), with no post-processing. Writes ``<output-dir>/aasce_test.csv``
(``image_id,landmark_id,x,y`` in 752 x 256 pixels) and
``<output-dir>/metrics.json``.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate  # noqa: E402
from pix2point.config import CONFIG, SCHEMA_PATH  # noqa: E402
from pix2point.dataset import SpineDataset, load_schema, subject_key  # noqa: E402
from pix2point.model import Pix2Point, argmax_coordinates  # noqa: E402

CHECKPOINT_SUFFIXES = {"best_mre": "_best_mre", "best_loss": "", "last": "_last"}
PROTOTYPE_HEAD_KEYS = ("num_prototypes", "prototype_mode", "proto_temperature_mode",
                       "proto_temperature", "proto_temperature_per_landmark")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="AASCE data root, containing test/{images_prepro_clahe,labels,cropping_info}")
    ap.add_argument("--weights-dir", required=True, help="folder holding the two weight files")
    ap.add_argument("--checkpoint", default="best_mre", choices=list(CHECKPOINT_SUFFIXES))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    args = ap.parse_args()

    print("Test run at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    suffix = CHECKPOINT_SUFFIXES[args.checkpoint]
    backbone = os.path.join(args.weights_dir, f"unet_backbone{suffix}.pth")
    prototypes = os.path.join(args.weights_dir, f"prototype_layer{suffix}.pth")
    for path in (backbone, prototypes):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint '{args.checkpoint}' not found: {path}")
    os.makedirs(args.output_dir, exist_ok=True)

    schema = load_schema(SCHEMA_PATH)
    cfg = copy.deepcopy(CONFIG)
    # The prototype head must be built as in training: read its settings from
    # the run's config.json when present (written by train.py).
    run_config = os.path.join(args.weights_dir, "config.json")
    if os.path.isfile(run_config):
        with open(run_config, encoding="utf-8") as fh:
            saved = json.load(fh)
        cfg.update({k: saved[k] for k in PROTOTYPE_HEAD_KEYS if k in saved})
        print(f"Prototype head settings read from {run_config}")
    print(f"Prototype head: prototype_mode={cfg['prototype_mode']}, "
          f"proto_temperature_mode={cfg['proto_temperature_mode']}")
    cfg.update(n_classes=len(schema), schema=schema,
               weights_backbone_load_path=backbone, weights_prototypes_load_path=prototypes)

    dataset = SpineDataset(args.data_root, schema, (cfg["target_height"], cfg["target_width"]), "test",
                           cfg["images_dirname"], cfg["labels_dirname"], None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test subjects: {len(dataset)}")

    model = Pix2Point(cfg)
    model.model.eval()
    model.finetuning_layer.eval()

    predictions = {}
    global_idx = 0
    with torch.no_grad():
        for imgs, _ in loader:
            heatmaps = model.predict_heatmaps(imgs.to(model.device, non_blocking=True))
            x_pred, y_pred = argmax_coordinates(heatmaps)
            x_pred, y_pred = x_pred.float().cpu(), y_pred.float().cpu()
            for b in range(heatmaps.shape[0]):
                subj = subject_key(dataset.samples[global_idx][0])
                global_idx += 1
                predictions[subj] = np.array(
                    [[round(float(x_pred[b, c]), 2), round(float(y_pred[b, c]), 2)]
                     for c in range(len(schema))], dtype=np.float64)
            print(f"  processed {global_idx}/{len(dataset)} subjects", flush=True)

    csv_path = os.path.join(args.output_dir, "aasce_test.csv")
    evaluate.write_predictions_csv(csv_path, predictions)
    print(f"Predictions: {csv_path}")

    res = evaluate.evaluate(args.data_root, csv_path)
    res["weights"] = {"checkpoint": args.checkpoint, "backbone": os.path.abspath(backbone),
                      "prototypes": os.path.abspath(prototypes)}
    print(evaluate.format_report(res))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"Metrics: {os.path.join(args.output_dir, 'metrics.json')}")
    print("End at:", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
