"""
pix2point: nnU-Net backbone + prototype head, trained with a pixel-wise
cross-entropy over each landmark heatmap.

The backbone returns a 512-channel feature map at full resolution. For every
landmark, the prototype head computes the cosine similarity between the
L2-normalised features of each pixel and the landmark prototype, divided by a
temperature; the resulting map, read as the logits of a softmax over the
H x W pixels, is the landmark heatmap. The prediction is its argmax. The
prototypes are either trained parameters or, in ``minibatch`` mode, the
embeddings of the landmarks in other images of the batch (k-Means centroids
over the training split outside training); see ``prototype_mode`` and
``proto_temperature_mode`` in :mod:`pix2point.config`.

The backbone can first be pretrained without labels (``run_pretraining``):
Barlow Twins loss between the projected pixel features of two augmented views
of an image, aligned by a known random affine transform.
"""

import math
import os

import einops
import matplotlib
matplotlib.use("Agg")               # headless backend (compute server)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from pix2point.dataset import inverse_resize_scales, non_augmented_loader
from pix2point.unet_nnunet import UNet_nnunet

# ``GradScaler`` moved to ``torch.amp`` in torch >= 2.4 (with a ``device``
# argument); on torch 2.0-2.3 it lives in ``torch.cuda.amp`` without that
# argument. This shim works on both.
try:
    from torch.amp import GradScaler as _GradScaler

    def make_grad_scaler(device_type: str):
        """Create a ``GradScaler`` enabled only on CUDA (torch >= 2.4 API)."""
        return _GradScaler(device_type, enabled=(device_type == "cuda"))
except ImportError:  # torch < 2.4
    from torch.cuda.amp import GradScaler as _GradScaler

    def make_grad_scaler(device_type: str):
        """Create a ``GradScaler`` enabled only on CUDA (legacy API)."""
        return _GradScaler(enabled=(device_type == "cuda"))


IGNORE_INDEX = -100

PROTOTYPE_MODES = ("trainable", "minibatch")
TEMPERATURE_MODES = ("unnormalized", "fixed", "learned")

# Logit given to absent prototype slots (landmark missing in a prototype
# image): far below any cosine logit, yet finite so that a fully masked
# landmark yields a flat -- not NaN -- softmax.
MASKED_LOGIT = -1.0e4

KMEANS_N_ITERS = 100


class PrototypeLayer(nn.Module):
    """Cosine-similarity prototype head: one heatmap of logits per landmark.

    The logits are ``cos(f, k_l) / tau_l``. The temperature is stored as
    ``log_temperature`` (keeps ``tau > 0``), of shape ``(num_classes,)`` when
    ``temperature_per_landmark`` else ``(1,)``, and is trained when
    ``learnable_temperature``. With ``normalize_prototypes=False`` the
    prototypes are not L2-normalised, so that their norm acts as a learned
    per-landmark inverse temperature (used with a fixed ``tau = 1``).
    """

    # Lower bound on tau: caps the logits at 1 / MIN_TEMPERATURE.
    MIN_TEMPERATURE = 0.01

    def __init__(self, in_channels, num_classes, num_prototypes=1, temperature=1.0,
                 learnable_temperature=False, temperature_per_landmark=False,
                 normalize_prototypes=False, init_scale=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.learnable_temperature = learnable_temperature
        self.temperature_per_landmark = temperature_per_landmark
        self.normalize_prototypes = normalize_prototypes
        n_temp = num_classes if temperature_per_landmark else 1
        self.log_temperature = nn.Parameter(torch.full((n_temp,), math.log(temperature)),
                                            requires_grad=learnable_temperature)
        self.prototypes = nn.Parameter(torch.randn(num_classes * num_prototypes, in_channels) * init_scale)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A fixed temperature is set by the configuration, not by the checkpoint
        # (checkpoints of the unnormalized mode may have no log_temperature).
        if not self.learnable_temperature:
            state_dict[prefix + 'log_temperature'] = self.log_temperature.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def temperature(self):
        """Per-landmark temperature ``(num_classes,)`` (expanded if shared)."""
        return self.log_temperature.clamp(min=math.log(self.MIN_TEMPERATURE)).exp().expand(self.num_classes)

    def forward(self, features, prototypes=None, prototype_mask=None):
        """Per-landmark logits ``(B, num_classes, H, W)``.

        ``prototypes`` ``(num_classes * num_prototypes, C)``, landmark-major,
        replaces ``self.prototypes`` (minibatch-dependent prototypes, always
        L2-normalised). ``prototype_mask`` ``(num_classes, num_prototypes)``
        bool: ``False`` slots are excluded from the logsumexp.
        """
        B, C, H, W = features.shape

        features = F.normalize(features, p=2, dim=1)                # (B, C, H, W)
        if prototypes is None:
            prototypes = self.prototypes
            if self.normalize_prototypes:
                prototypes = F.normalize(prototypes, p=2, dim=1)    # (num_classes * K, C)
        else:
            prototypes = F.normalize(prototypes, p=2, dim=1)

        features_flat = features.view(B, C, -1)                     # (B, C, H*W)
        logits = torch.einsum('b c n, k c -> b k n', features_flat, prototypes)

        raw_logits = logits.view(B, self.num_classes, self.num_prototypes, H, W)
        raw_logits = raw_logits / self.temperature.view(1, self.num_classes, 1, 1, 1)
        if prototype_mask is not None:
            raw_logits = raw_logits.float().masked_fill(
                ~prototype_mask.view(1, self.num_classes, self.num_prototypes, 1, 1), MASKED_LOGIT)
        # Aggregate the K prototypes of a landmark (identity for K = 1).
        return torch.logsumexp(raw_logits, dim=2)                   # (B, num_classes, H, W)


def build_prototype_layer(cfg, in_channels):
    """``PrototypeLayer`` of the ``prototype_mode`` / ``proto_temperature_mode`` of ``cfg``."""
    prototype_mode = cfg["prototype_mode"]
    temperature_mode = cfg["proto_temperature_mode"]
    if prototype_mode not in PROTOTYPE_MODES:
        raise ValueError(f"prototype_mode must be one of {PROTOTYPE_MODES}, got {prototype_mode!r}")
    if temperature_mode not in TEMPERATURE_MODES:
        raise ValueError(f"proto_temperature_mode must be one of {TEMPERATURE_MODES}, got {temperature_mode!r}")
    if prototype_mode == "minibatch" and temperature_mode == "unnormalized":
        raise ValueError("Minibatch prototypes are L2-normalised: use proto_temperature_mode "
                         "'fixed' or 'learned'.")
    unnormalized = temperature_mode == "unnormalized"
    learned = temperature_mode == "learned"
    return PrototypeLayer(
        in_channels, int(cfg["n_classes"]), num_prototypes=int(cfg["num_prototypes"]),
        temperature=1.0 if unnormalized else float(cfg["proto_temperature"]),
        learnable_temperature=learned,
        temperature_per_landmark=learned and bool(cfg["proto_temperature_per_landmark"]),
        normalize_prototypes=not unnormalized,
        init_scale=float(cfg["proto_init_scale"]),
    )


def gather_landmark_embeddings(features, targets):
    """L2-normalised embeddings ``(B, L, D)`` at the ground-truth pixels, and
    their validity ``(B, L)`` (embeddings of absent landmarks are meaningless)."""
    B, D = features.shape[:2]
    valid = targets != IGNORE_INDEX
    idx = torch.where(valid, targets, torch.zeros_like(targets)).long()
    emb = torch.gather(features.reshape(B, D, -1), 2, idx.unsqueeze(1).expand(B, D, idx.shape[1]))
    return F.normalize(emb.transpose(1, 2), p=2, dim=2), valid


def spherical_kmeans(x, k, n_iters=KMEANS_N_ITERS, n_init=10, generator=None):
    """k-Means with cosine similarity on unit-norm points ``x`` ``(N, D)``.

    k-means++ seeding, restarted ``n_init`` times; the run with the highest
    mean cosine similarity between each point and its centroid is kept.
    Returns unit-norm centroids ``(k, D)`` (the points, cycled, if ``N <= k``).
    """
    N = x.shape[0]
    if N <= k:
        return x[torch.arange(k, device=x.device) % N].clone()

    best_centroids, best_score = None, -math.inf
    for _ in range(max(1, int(n_init))):
        centroids = _spherical_kmeans_single_run(x, k, n_iters, generator)
        score = (x @ centroids.t()).max(dim=1).values.mean().item()
        if score > best_score:
            best_centroids, best_score = centroids, score
    return best_centroids


def _spherical_kmeans_single_run(x, k, n_iters, generator):
    """One k-means++ seeding followed by Lloyd iterations (requires ``N > k``)."""
    N = x.shape[0]
    first = torch.randint(N, (1,), generator=generator, device=x.device)
    centroids = x[first]
    for _ in range(1, k):
        # For unit vectors, squared Euclidean distance = 2 * (1 - cosine).
        dist = (1.0 - (x @ centroids.t()).max(dim=1).values).clamp(min=0)
        if dist.sum() <= 0:
            nxt = torch.randint(N, (1,), generator=generator, device=x.device)
        else:
            nxt = torch.multinomial(dist, 1, generator=generator)
        centroids = torch.cat([centroids, x[nxt]], dim=0)

    for _ in range(n_iters):
        assign = (x @ centroids.t()).argmax(dim=1)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        empty = torch.bincount(assign, minlength=k) == 0
        if empty.any():
            # Re-seed empty clusters with the points worst fitted by their centroid.
            fit = (x * centroids[assign]).sum(dim=1)
            sums[empty] = x[fit.argsort()[: int(empty.sum())]]
        new_centroids = F.normalize(sums, p=2, dim=1)
        converged = torch.allclose(new_centroids, centroids, atol=1e-6)
        centroids = new_centroids
        if converged:
            break
    return centroids


class MLP(nn.Module):
    """1x1-convolution projection head (pretraining only)."""

    def __init__(self, channels_in, channels_out, inner_dim_1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, inner_dim_1, 1),
            nn.BatchNorm2d(inner_dim_1),
            nn.ReLU(),
            nn.Conv2d(inner_dim_1, channels_out, 1),
        )

    def forward(self, x):
        return self.net(x)


def generate_affine_spatial_transform(batch_size, is_rotated=True, is_cropped=True, is_flipped=True,
                                      is_translation=False, max_angle=np.pi / 2, max_crop=0.5):
    """Random ``(B, 2, 3)`` affine matrices (normalised coordinates) for the pretraining.

    rotation x crop (scale in (max_crop, 0.95) + shift inside the frame) x
    translation (half of the batch) x flip (per sample: mirror along x, along
    y, or none).
    """
    identity_affine = torch.eye(3).float()
    identity_affine = identity_affine.repeat((batch_size, 1, 1))

    zeros = torch.zeros(batch_size)
    ones = torch.ones(batch_size)

    if is_rotated:
        theta = torch.FloatTensor(batch_size).uniform_(-max_angle, max_angle)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        rotation = torch.stack((torch.stack([cos_theta, -sin_theta, zeros], dim=-1),
                                torch.stack([sin_theta, cos_theta, zeros], dim=-1),
                                torch.stack([zeros, zeros, ones], dim=-1)), dim=1).float()
    else:
        rotation = identity_affine.detach().clone()

    if is_cropped:
        scale_factor = torch.FloatTensor(batch_size).uniform_(max_crop, 0.95)
        translate_height = 2 * (1 - scale_factor) * torch.rand(batch_size) - (1 - scale_factor)
        translate_width = 2 * (1 - scale_factor) * torch.rand(batch_size) - (1 - scale_factor)
        crop = torch.stack((torch.stack([scale_factor, zeros, translate_width], dim=-1),
                            torch.stack([zeros, scale_factor, translate_height], dim=-1),
                            torch.stack([zeros, zeros, ones], dim=-1)), dim=1).float()
    else:
        crop = identity_affine.detach().clone()

    if is_flipped:
        horizontal_flip = torch.tensor([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]]).float()
        flip = torch.tensor([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]).float()
        flip = flip.repeat((batch_size, 1, 1))
        random_indexes = torch.randint(batch_size, (int(batch_size * 0.5), ))
        flip[random_indexes] = horizontal_flip
        random_indexes = torch.randint(batch_size, (int(batch_size * 0.3), ))
        flip[random_indexes] = torch.eye(3)
    else:
        flip = identity_affine.detach().clone()

    if is_translation:
        dx = torch.FloatTensor(batch_size).uniform_(-1, 1)
        dy = torch.FloatTensor(batch_size).uniform_(-1, 1)
        translation = torch.stack((torch.stack([ones, zeros, dx], dim=-1),
                                   torch.stack([zeros, ones, dy], dim=-1),
                                   torch.stack([zeros, zeros, ones], dim=-1)), dim=1).float()
        random_indexes = torch.randint(batch_size, (int(batch_size * 0.5), ))
        translation[random_indexes] = torch.eye(3)
    else:
        translation = identity_affine.detach().clone()

    final_affine_matrix = torch.bmm(rotation, torch.bmm(crop, torch.bmm(translation, flip)))
    return final_affine_matrix[:, :-1, :]


def off_diagonal(x):
    """Flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(features_1, features_2, lambda_param, device):
    """Barlow Twins loss between two ``(B, C, H, W)`` maps, every pixel being a sample."""
    features_1 = einops.rearrange(features_1, 'b c h w -> (b h w) c').float()
    features_2 = einops.rearrange(features_2, 'b c h w -> (b h w) c').float()
    N = features_1.shape[0]

    bn = torch.nn.BatchNorm1d(features_1.shape[1], affine=False, track_running_stats=False).to(device)

    with autocast("cuda", enabled=False):
        features_1 = bn(features_1)
        features_2 = bn(features_2)

        c = features_1.T @ features_2
        c.div_(N)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambda_param * off_diag

    return loss


def argmax_coordinates(heatmaps: torch.Tensor):
    """``(x, y)`` integer pixel of the maximum of each ``(B, C, H, W)`` heatmap."""
    B, C, H, W = heatmaps.shape
    pred_idx = heatmaps.view(B, C, -1).argmax(dim=2)
    y = torch.div(pred_idx, W, rounding_mode="floor")
    x = pred_idx % W
    return x, y


class Pix2Point:
    """Backbone + prototype head, with the finetuning loop of the protocol.

    ``cfg`` is :data:`pix2point.config.CONFIG` completed with ``n_classes``,
    ``schema``, ``train_dir`` and, optionally, ``weights_backbone_load_path``
    (pretrained backbone) / ``weights_prototypes_load_path`` (inference).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            # Fixed cuDNN algorithms: benchmark autotuning picks them by timing,
            # which makes results differ from one run to the next.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        n_features_map = cfg["n_features_map"]
        self.model = UNet_nnunet(in_channels=cfg["n_channels"],
                                 n_features_map=n_features_map).to(self.device)

        self.prototype_mode = cfg["prototype_mode"]
        self.finetuning_layer = build_prototype_layer(cfg, n_features_map).to(self.device)
        if self.prototype_mode == "minibatch":
            # Drawn from the batch during training: the stored prototypes only
            # hold the k-Means centroids used outside training.
            self.finetuning_layer.prototypes.requires_grad_(False)
        self.finetune_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(self.device)

        # Used by the pretraining only. Also built for the supervised training,
        # after the prototypes as in the reference runs, so that a given seed
        # draws the same initial weights and data-loader seeds (batch order).
        self.projection_head = MLP(channels_in=n_features_map,
                                   channels_out=cfg["n_features_map_mlp"],
                                   inner_dim_1=cfg["inner_dim_1"]).to(self.device)

        backbone_path = cfg.get("weights_backbone_load_path")
        if backbone_path:
            self.model.load_state_dict(torch.load(backbone_path, map_location=self.device))
            print(f"Backbone loaded from {backbone_path}.")

        prototypes_path = cfg.get("weights_prototypes_load_path")
        if prototypes_path:
            self.finetuning_layer.load_state_dict(torch.load(prototypes_path, map_location=self.device))
            print(f"Prototype layer loaded from {prototypes_path}.")

    def compute_pretraining_loss(self, view_1, view_2):
        """Barlow Twins loss between the two views, aligned by a random affine transform."""
        cfg = self.cfg
        transfo_affine = generate_affine_spatial_transform(
            view_1.shape[0], cfg["is_rotated"], cfg["is_cropped"], cfg["is_flipped"],
            cfg["is_translation"], cfg["max_angle"], cfg["max_crop"],
        ).to(self.device)

        # Warp the second view...
        grid = F.affine_grid(transfo_affine, view_2.shape, align_corners=False)
        view_2 = F.grid_sample(view_2, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        features_1 = self.projection_head(self.model(view_1))
        features_2 = self.projection_head(self.model(view_2))

        # ...and the features of the first, so that pixels correspond.
        features_1 = F.grid_sample(features_1, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return barlow_loss(features_1, features_2, cfg["lambda_barlow"], self.device)

    def run_pretraining(self, training_loader, validation_loader):
        """Pretrain backbone + projection head for ``num_epochs`` (no early stopping).

        Writes ``unet_backbone_last.pth`` every epoch and ``unet_backbone_best.pth``
        (lowest validation loss) in ``cfg['train_dir']``.
        """
        cfg = self.cfg
        num_epochs = cfg["num_epochs"]
        save_path_best = os.path.join(cfg["train_dir"], "unet_backbone_best.pth")
        save_path_last = os.path.join(cfg["train_dir"], "unet_backbone_last.pth")
        avg_train_losses, avg_val_losses = [], []

        scaler = make_grad_scaler(self.device.type)
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.projection_head.parameters()),
                                     lr=cfg["learning_rate_backbone"])
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss, val_loss = [], []

            self.model.train()
            self.projection_head.train()
            for batch in training_loader:
                view_1 = batch[0].float().to(self.device, non_blocking=True)
                view_2 = batch[1].float().to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type):
                    batch_loss_training = self.compute_pretraining_loss(view_1, view_2)
                train_loss.append(batch_loss_training.item())

                scaler.scale(batch_loss_training).backward()
                scaler.step(optimizer)
                scaler.update()

            avg_train_loss = np.average(train_loss)
            avg_train_losses.append(avg_train_loss)

            self.model.eval()
            self.projection_head.eval()
            with torch.no_grad():
                for batch_val in validation_loader:
                    view_1 = batch_val[0].float().to(self.device, non_blocking=True)
                    view_2 = batch_val[1].float().to(self.device, non_blocking=True)
                    val_loss.append(self.compute_pretraining_loss(view_1, view_2).item())

                avg_val_loss = np.average(val_loss)
                avg_val_losses.append(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), save_path_best)
                torch.save(self.model.state_dict(), save_path_last)

            print(f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] | Train Loss: {avg_train_loss:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f}", flush=True)

        return avg_train_losses, avg_val_losses

    def predict_heatmaps(self, inputs):
        """Landmark logits ``(B, n_classes, H, W)`` (no autocast, as at test time)."""
        return self.finetuning_layer(self.model(inputs))

    def compute_finetuning_loss(self, inputs, targets, minibatch_prototypes=False, distance_weights=None):
        """Pixel-wise CrossEntropy over the H x W pixels of each landmark heatmap.

        ``minibatch_prototypes`` draws the prototypes from the batch itself
        (training, minibatch mode). With ``distance_weights`` ``(B,)``, also
        returns the sum of the argmax distances, each multiplied by its
        sample's weight, and the number of points: ``(loss, dist_sum, n)``.
        """
        features = self.model(inputs)
        proto_kwargs = {}
        if minibatch_prototypes:
            features, targets, prototypes, prototype_mask = self._split_minibatch_prototypes(features, targets)
            proto_kwargs = {"prototypes": prototypes, "prototype_mask": prototype_mask}
        predictions = self.finetuning_layer(features, **proto_kwargs)

        B, num_points, H, W = predictions.shape
        loss_ce = self.finetune_loss_fn(predictions.reshape(B * num_points, H * W),
                                        targets.reshape(B * num_points))
        if distance_weights is None:
            return loss_ce

        with torch.no_grad():
            pred_idx = torch.argmax(predictions.view(B, num_points, -1), dim=2)
            y_pred = torch.div(pred_idx, W, rounding_mode='floor')
            x_pred = pred_idx % W

            mask = targets != IGNORE_INDEX
            y_gt = torch.div(targets, W, rounding_mode='floor')
            x_gt = targets % W

            dist = torch.sqrt((x_gt[mask].float() - x_pred[mask].float())**2
                              + (y_gt[mask].float() - y_pred[mask].float())**2)
            weights = distance_weights.to(dist.device, torch.float64).view(B, 1).expand(B, num_points)[mask]
        return loss_ce, (dist.double() * weights).sum().item(), int(mask.sum().item())

    def _split_minibatch_prototypes(self, features, targets):
        """Draw minibatch-dependent prototypes and the remaining query images.

        ``P' = min(P, B - 1)`` random images of the batch provide the prototypes
        ``k_{l,p} = f(x^p)(y_l^p)`` (gradients flow through them) and are
        excluded from the loss. Slots of landmarks absent from a prototype
        image are masked; a landmark with no valid slot is ignored in the loss.

        Returns ``(query_features, query_targets, prototypes (L * P, D),
        prototype_mask (L, P))``.
        """
        B = features.shape[0]
        L = self.finetuning_layer.num_classes
        P = self.finetuning_layer.num_prototypes
        n_proto = min(P, B - 1)
        if n_proto < 1:
            raise ValueError("Minibatch-dependent prototypes need at least 2 images per batch.")

        perm = torch.randperm(B, device=features.device)
        proto_idx, query_idx = perm[:n_proto], perm[n_proto:]
        emb, valid = gather_landmark_embeddings(features[proto_idx], targets[proto_idx])  # (P', L, D), (P', L)

        D = emb.shape[-1]
        prototypes = emb.transpose(0, 1)                                     # (L, P', D)
        mask = valid.t()                                                     # (L, P')
        if n_proto < P:
            prototypes = torch.cat([prototypes, prototypes.new_zeros(L, P - n_proto, D)], dim=1)
            mask = torch.cat([mask, mask.new_zeros(L, P - n_proto)], dim=1)

        has_proto = mask.any(dim=1).unsqueeze(0)                             # (1, L)
        query_targets = targets[query_idx].masked_fill(~has_proto, IGNORE_INDEX)
        return features[query_idx], query_targets, prototypes.reshape(L * P, D), mask

    @torch.no_grad()
    def update_kmeans_prototypes(self, loader):
        """Set the stored prototypes to the per-landmark spherical k-Means
        centroids (k = num_prototypes) of the ground-truth embeddings over
        ``loader`` (eval mode). Landmarks never annotated keep their prototypes."""
        was_training = self.model.training
        self.model.eval()

        all_emb, all_valid = [], []
        for batch in loader:
            inputs = batch[0].to(self.device, non_blocking=True)
            targets = batch[1].long().to(self.device, non_blocking=True)
            with autocast(device_type=self.device.type):
                features = self.model(inputs)
            emb, valid = gather_landmark_embeddings(features.float(), targets)
            all_emb.append(emb.cpu())
            all_valid.append(valid.cpu())
        emb = torch.cat(all_emb)        # (N, L, D)
        valid = torch.cat(all_valid)    # (N, L)

        layer = self.finetuning_layer
        L, P = layer.num_classes, layer.num_prototypes
        centroids = layer.prototypes.detach().float().cpu().view(L, P, -1).clone()
        generator = torch.Generator().manual_seed(int(self.cfg["seed"]))
        missing = []
        for l in range(L):
            points = emb[valid[:, l], l]
            if points.shape[0] == 0:
                missing.append(l)
                continue
            centroids[l] = spherical_kmeans(points, P, n_init=int(self.cfg["kmeans_n_init"]),
                                            generator=generator)
        layer.prototypes.data.copy_(centroids.view(L * P, -1).to(layer.prototypes))

        if missing:
            print(f"k-Means prototypes: no annotation for classes {missing} -> previous prototypes kept.")
        if was_training:
            self.model.train()

    def _resolve_viz_landmarks(self, name_pairs):
        """Class indices and labels of the ``(vertebra, corner)`` names in the schema."""
        index = {tuple(pair): i for i, pair in enumerate(self.cfg["schema"])}
        indices, labels = [], []
        for group, point in name_pairs:
            if (group, point) in index:
                indices.append(index[(group, point)])
                labels.append(f"{group}.{point}")
            else:
                print(f"[viz] landmark {(group, point)} not in the schema -> skipped")
        return indices, labels

    def _save_final_heatmaps(self, inputs, targets, indices, labels, epoch, out_dir):
        """Save the heatmaps of a fixed validation image (pred in red, GT in green)."""
        was_training = self.model.training
        self.model.eval()
        self.finetuning_layer.eval()
        with torch.no_grad():
            preds = self.finetuning_layer(self.model(inputs))   # (1, C, H, W)
            num_classes = preds.shape[1]
            probs = torch.softmax(preds.view(1, num_classes, -1), dim=2).view_as(preds)

        img = inputs[0, 0].float().cpu().numpy()
        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 8))
        if n == 1:
            axes = [axes]
        W = img.shape[1]
        for ax, idx, lab in zip(axes, indices, labels):
            heat = probs[0, idx].cpu().numpy()
            y, x = np.unravel_index(int(heat.argmax()), heat.shape)
            ax.imshow(img, cmap='gray')
            ax.imshow(heat, cmap='jet', alpha=0.5)
            ax.scatter([x], [y], c='red', s=25, marker='x', label='pred')
            gt = int(targets[0, idx].item())
            if gt != -100:
                ax.scatter([gt % W], [gt // W], c='lime', s=40, marker='+', label='GT')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
            ax.set_title(lab, fontsize=12)
            ax.axis('off')
        fig.suptitle(f"Epoch {epoch + 1}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch + 1:03d}.png"))
        plt.close(fig)

        if was_training:
            self.model.train()
            self.finetuning_layer.train()

    def run_finetuning(self, training_loader, validation_loader):
        """Train until early stopping on the validation MRE.

        The validation MRE is measured in pixels of the original radiograph.
        Three checkpoints are written in ``cfg['train_dir']``:
        ``*_best_mre.pth`` (best validation MRE, the reported results),
        ``unet_backbone.pth`` / ``prototype_layer.pth`` (best validation loss)
        and ``*_last.pth`` (last epoch). In minibatch mode, the stored
        prototypes are the k-Means centroids of the current backbone, updated
        before each validation.
        """
        cfg = self.cfg
        train_dir = cfg["train_dir"]
        num_epochs = cfg["num_epochs"]
        avg_train_losses, avg_val_losses, avg_val_distances = [], [], []
        layer = self.finetuning_layer

        minibatch_mode = self.prototype_mode == "minibatch"
        centroid_loader = None
        if minibatch_mode:
            P = layer.num_prototypes
            if training_loader.batch_size <= P:
                raise ValueError(f"prototype_mode='minibatch' needs batch_size > num_prototypes "
                                 f"(got batch_size={training_loader.batch_size}, num_prototypes={P}).")
            centroid_loader = non_augmented_loader(training_loader)
            print(f"Prototypes: minibatch-dependent ({P} image(s) per batch); k-Means centroids "
                  f"(k={P}) over the training split for validation and the saved checkpoints.")
        else:
            print("Prototypes: trainable parameters.")
        print(f"Temperature: {cfg['proto_temperature_mode']}"
              + (", per landmark" if layer.temperature_per_landmark else "")
              + f" (tau init = {layer.temperature[0].item():.4g}"
              + (", prototypes not normalised)" if not layer.normalize_prototypes else ")"))

        scaler = make_grad_scaler(self.device.type)
        optimizers = [torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate_backbone"])]
        # Prototypes and / or temperature, if trained.
        head_params = [p for p in layer.parameters() if p.requires_grad]
        if head_params:
            optimizers.append(torch.optim.Adam(head_params, lr=cfg["learning_rate_prototypes"]))

        paths = {
            name: (os.path.join(train_dir, f"unet_backbone{suffix}.pth"),
                   os.path.join(train_dir, f"prototype_layer{suffix}.pth"))
            for name, suffix in (("best_loss", ""), ("best_mre", "_best_mre"), ("last", "_last"))
        }

        def save(name):
            torch.save(self.model.state_dict(), paths[name][0])
            torch.save(layer.state_dict(), paths[name][1])

        best_val_loss = float('inf')
        best_val_mre = float('inf')
        patience_counter = 0
        early_stopping_patience = int(cfg["early_stopping_patience"])
        val_inverse_scales = inverse_resize_scales(validation_loader)
        print(f"Early stopping on the validation MRE (original image px), patience {early_stopping_patience}")

        # Heatmaps of the same validation image (first sample; the validation
        # loader is not shuffled), saved every epoch.
        viz_indices, viz_labels = self._resolve_viz_landmarks(cfg["viz_landmarks"])
        _viz_batch = next(iter(validation_loader))
        viz_inputs = _viz_batch[0][:1].to(self.device, non_blocking=True)
        viz_targets = _viz_batch[1][:1].long()   # (1, n_classes) ground-truth flat indices
        heatmap_final_dir = os.path.join(train_dir, 'heatmaps_final')
        os.makedirs(heatmap_final_dir, exist_ok=True)
        print(f"Heatmap tracking: {viz_labels} -> classes {viz_indices} "
              f"(saved every epoch in {heatmap_final_dir})")

        for epoch in range(num_epochs):
            train_loss = []
            val_loss = []

            self.model.train()
            layer.train()

            print(f"\n=== Epoch {epoch + 1}/{num_epochs} | Training ({len(training_loader)} batches) ===",
                  flush=True)

            for batch in training_loader:
                inputs = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].long().to(self.device, non_blocking=True)
                if minibatch_mode and inputs.shape[0] < 2:
                    continue   # no query image left once the prototype image is drawn

                for o in optimizers:
                    o.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type):
                    batch_loss_training = self.compute_finetuning_loss(
                        inputs, targets, minibatch_prototypes=minibatch_mode)

                train_loss.append(batch_loss_training.item())

                scaler.scale(batch_loss_training).backward()
                for o in optimizers:
                    scaler.step(o)
                scaler.update()

            avg_train_loss = np.average(train_loss)
            avg_train_losses.append(avg_train_loss)
            if layer.learnable_temperature:
                tau = layer.temperature.detach().float()
                print(f"Temperature: mean={tau.mean().item():.4f} min={tau.min().item():.4f} "
                      f"max={tau.max().item():.4f}")

            # The centroids must match the current backbone before validating
            # and before any checkpoint of the prototype layer is written.
            if minibatch_mode:
                self.update_kmeans_prototypes(centroid_loader)
            save("last")

            self.model.eval()
            layer.eval()
            with torch.no_grad():
                n_val_batches = len(validation_loader)
                print(f"--- Epoch {epoch + 1}/{num_epochs} | Validation ({n_val_batches} batches) ---",
                      flush=True)
                val_offset, val_dist_sum, val_dist_n = 0, 0.0, 0
                for batch_index_val, batch_val in enumerate(validation_loader):
                    inputs = batch_val[0].to(self.device, non_blocking=True)
                    targets = batch_val[1].long().to(self.device, non_blocking=True)
                    weights = val_inverse_scales[val_offset:val_offset + inputs.shape[0]]
                    val_offset += inputs.shape[0]

                    with autocast(device_type=self.device.type):
                        batch_loss_validation, dist_sum, dist_n = self.compute_finetuning_loss(
                            inputs, targets, distance_weights=weights)

                    val_loss.append(batch_loss_validation.item())
                    val_dist_sum += dist_sum
                    val_dist_n += dist_n
                    batch_dist = dist_sum / dist_n if dist_n else 0.0
                    print(f"  [Val]   Epoch {epoch + 1}/{num_epochs} | Batch {batch_index_val + 1}/{n_val_batches} "
                          f"| val_loss={batch_loss_validation.item():.4f} | dist={batch_dist:.2f}px", flush=True)

                avg_val_loss = np.average(val_loss)
                avg_val_losses.append(avg_val_loss)
                avg_val_distance = val_dist_sum / val_dist_n if val_dist_n else 0.0
                avg_val_distances.append(avg_val_distance)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save("best_loss")

                if avg_val_distance < best_val_mre:
                    best_val_mre = avg_val_distance
                    patience_counter = 0
                    save("best_mre")
                    print(f"--> New best val MRE: {best_val_mre:.2f}px (epoch {epoch + 1}), weights saved")
                else:
                    patience_counter += 1

            print(f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] | Train Loss: {avg_train_loss:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f} | Val Dist: {avg_val_distance:.2f}px")

            if viz_indices:
                try:
                    self._save_final_heatmaps(viz_inputs, viz_targets, viz_indices, viz_labels,
                                              epoch, heatmap_final_dir)
                except Exception as e:
                    print(f"Could not save the tracking heatmaps: {e}")

            if patience_counter >= early_stopping_patience:
                print(f'--> Early stopping at epoch {epoch + 1} '
                      f'({early_stopping_patience} epochs without val MRE improvement)')
                break

        plt.figure(figsize=(10, 5))
        plt.plot(avg_val_distances, label='Mean validation distance (px)', color='orange', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Distance (px, original image)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(train_dir, "val_distance.png"))
        plt.close()

        return avg_train_losses, avg_val_losses
