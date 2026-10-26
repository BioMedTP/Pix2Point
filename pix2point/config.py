"""
Experimental protocol of the AASCE runs (training from scratch and from a
pretrained backbone). The two runs share every value below; they differ only
by the backbone initialisation, given to ``train.py`` on the command line.

Paths (data, outputs, weights) are not set here: they are command-line
arguments of ``train.py`` / ``predict.py`` / ``evaluate.py``.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)

# 68 landmarks: the 4 vertebral-body corners of T1..L5, in this order. The order
# defines the output channels of the model and the landmark_id of the CSVs.
SCHEMA_PATH = os.path.join(PACKAGE_DIR, "landmark_schema_aasce.json")
# Frozen train / val subject lists (grouped by patient, 424 / 47 images).
TRAIN_SPLIT_FILE = os.path.join(REPO_DIR, "splits", "aasce_train.txt")
VAL_SPLIT_FILE = os.path.join(REPO_DIR, "splits", "aasce_val.txt")

CONFIG = {
    "seed": 42,

    # ----- Data -----------------------------------------------------------
    # Images are already preprocessed (752 x 256, uint8, CLAHE): no resize,
    # no intensity windowing, no CLAHE in the loader.
    "images_dirname": "images_prepro_clahe",
    "labels_dirname": "labels",
    "target_height": 752,
    "target_width": 256,
    "batch_size": 8,
    "num_workers": 10,

    # ----- Model ----------------------------------------------------------
    "n_channels": 1,                     # grayscale input
    "n_features_map": 512,               # backbone output channels
    "num_prototypes": 1,                 # prototypes per landmark
    # How the landmark prototypes are obtained:
    #   "trainable" : parameters optimised end-to-end.
    #   "minibatch" : at each training step, the embeddings at the ground-truth
    #                 landmarks of num_prototypes images drawn from the batch
    #                 (these images are then excluded from the loss); for
    #                 validation, the saved checkpoints and inference, the
    #                 per-landmark spherical k-Means centroids (k = num_prototypes)
    #                 of the embeddings of the whole (non-augmented) training split.
    "prototype_mode": "trainable",
    "kmeans_n_init": 10,                 # k-Means restarts (minibatch mode)
    # Temperature tau of the cosine logits cos(f, k_l) / tau:
    #   "unnormalized" : prototypes NOT L2-normalised, tau = 1 fixed: the norm
    #                    of each prototype acts as a learned inverse temperature
    #                    (trainable prototypes only).
    #   "fixed"        : L2-normalised prototypes, tau = proto_temperature.
    #   "learned"      : L2-normalised prototypes, tau learned, initialised to
    #                    proto_temperature; one tau per landmark if
    #                    proto_temperature_per_landmark, else one shared tau.
    "proto_temperature_mode": "unnormalized", # "learned" for batch-dependent prototypes
    "proto_temperature": 1.0,            # "fixed" / "learned" only. '0.1' for batch-dependent prototypes
    "proto_temperature_per_landmark": True,   # "learned" only. True for batch-dependent prototypes
    "proto_init_scale": 1.0,             # std of the random prototype initialisation
    # Projection head: used by the pretraining only; built (unused) during the
    # supervised training to keep the random stream of the original runs
    # (see Pix2Point.__init__).
    "n_features_map_mlp": 128,
    "inner_dim_1": 128,

    # ----- Optimisation ---------------------------------------------------
    "num_epochs": 200,
    "learning_rate_backbone": 1.0e-4,     # Adam, backbone
    "learning_rate_prototypes": 1.0e-3,   # Adam, prototype layer
    # Early stopping on the validation MRE, measured in pixels of the original
    # radiograph (each subject's distances divided by its resize_scale, read
    # from train/cropping_info). The best-val-MRE (reported results),
    # best-val-loss and last-epoch weights are all saved.
    "early_stopping_patience": 40,

    # ----- Augmentation (training split only) -----------------------------
    # Geometric, each applied with probability geometric_p. No horizontal flip:
    # it would swap the left / right corners without swapping their labels.
    #   random_crop = False : one affine transform, rotation +/- max_rotation_deg,
    #                         translation +/- max_translation, zoom-in only with
    #                         a scale in (1, 1 + max_zoom).
    #   random_crop = True  : a random crop of 1 / (1 + max_zoom) to 100 % of the
    #                         image area (aspect ratio kept), resized back to
    #                         752 x 256 (random zoom-in), then a rotation of
    #                         +/- max_rotation_deg (no translation).
    "random_crop": False,
    "geometric_p": 0.5,
    "max_rotation_deg": 5.0,
    "max_translation": 0.02,
    "max_zoom": 0.5,
    # Intensity augmentations.
    "random_blur_p": 0.2,
    "random_blur_std": 1.0,
    "random_noise_p": 0.2,
    "random_noise_std": 0.05,
    "random_gamma_p": 0.5,
    "random_gamma_limit": 20.0,          # gamma (%) in (100 - x, 100 + x)
    "random_brightness_contrast_p": 0.5,
    "random_brightness_limit": 0.15,     # brightness factor in (-x, x)
    "random_contrast_limit": 0.15,       # contrast factor in (-x, x)

    # Landmarks whose heatmap on the first validation image is saved every
    # epoch (<output_dir>/train/heatmaps_final/).
    "viz_landmarks": [
        ["Vertebre_T1", "Plat_Sup_D"],
        ["Vertebre_T8", "Plat_Sup_G"],
        ["Vertebre_L5", "Plat_Inf_D"],
    ],
}

# Self-supervised pretraining of the backbone (pretrain.py): no labels, same
# frozen train / val split (the 47 validation images are never trained on).
# Two views of each image (independent intensity augmentations); the second
# view and the features of the first are warped by the same random affine
# transform, and a Barlow Twins loss is applied to the projected features of
# every pixel. 200 epochs, no early stopping; the last-epoch backbone
# initialises the supervised training.
PRETRAIN_CONFIG = {
    **CONFIG,
    "batch_size": 4,
    "num_workers": 4,
    "num_epochs": 200,
    "learning_rate_backbone": 5.0e-4,     # Adam, backbone + projection head
    "n_features_map_mlp": 256,            # projection head output channels
    "inner_dim_1": 256,                   # projection head hidden channels
    "lambda_barlow": 5.0e-3,              # weight of the off-diagonal term
    # Random affine transform between the two views (normalised coordinates):
    # rotation in +/- max_angle rad, scale in (max_crop, 0.95) with a random
    # shift inside the frame, translation in [-1, 1] for half of the batch,
    # flips (horizontal / vertical / none) drawn per sample.
    "is_rotated": True,
    "is_cropped": True,
    "is_flipped": False,
    "is_translation": False,
    "max_angle": 0.2,
    "max_crop": 0.4,
    # Intensity augmentations of each view.
    "random_blur_p": 0.2,
    "random_blur_std": 1.0,
    "random_noise_p": 0.25,
    "random_noise_std": 0.1,
    "random_gamma_p": 0.5,
    "random_gamma_limit": 20.0,
    "random_brightness_contrast_p": 0.5,
    "random_brightness_limit": 0.2,
    "random_contrast_limit": 0.2,
}
