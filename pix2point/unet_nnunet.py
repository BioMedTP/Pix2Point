"""
Reproduction of the nnU-Net v2 ``PlainConvUNet`` (2D configuration planned by
nnU-Net for 752 x 256 inputs), used by pix2point as a feature extractor.

A plain (non-residual) U-Net with InstanceNorm + LeakyReLU, strided
convolutions for downsampling and transposed convolutions for upsampling:

    ConvNormAct      one Conv -> Norm -> Activation   (the atomic block)
    StackedConvBlock N convs, the first one carries the stride
    EncoderStage     a stage of the contracting path
    DecoderStage     transpose-conv + skip concat + convs
    OutConv          1x1 segmentation head
    UNet_nnunet      assembles everything

The segmentation heads (``seg_layer`` / ``output_head``) are not used in
feature mode but are kept: they are part of the released checkpoints'
state dicts and of the weight-initialisation order.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Architecture planned by nnU-Net (configuration "2d") ---------------------
_FEATURES_PER_STAGE: Tuple[int, ...] = (32, 64, 128, 256, 512, 512, 512)
_KERNEL_SIZES: Tuple[Tuple[int, int], ...] = ((3, 3),) * 7
_STRIDES: Tuple[Tuple[int, int], ...] = ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2))
_N_CONV_PER_STAGE: Tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2)
_N_CONV_PER_STAGE_DECODER: Tuple[int, ...] = (2, 2, 2, 2, 2, 2)
_CONV_BIAS: bool = True
_NORM_KWARGS = {"eps": 1e-5, "affine": True}
_NONLIN_KWARGS = {"inplace": True}


class ConvNormAct(nn.Module):
    """Atomic block: ``Conv2d -> InstanceNorm2d -> LeakyReLU``.

    nnU-Net's ``ConvDropoutNormReLU`` without dropout. The convolution carries
    the stride, so a strided block both convolves and downsamples.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        conv_bias: bool = _CONV_BIAS,
    ) -> None:
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)  # 'same' spatial size for stride 1
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=conv_bias,
        )
        self.norm = nn.InstanceNorm2d(out_channels, **_NORM_KWARGS)
        self.act = nn.LeakyReLU(**_NONLIN_KWARGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class StackedConvBlock(nn.Module):
    """A stage of ``num_convs`` convolutions (nnU-Net ``StackedConvBlocks``).

    Only the **first** convolution applies ``stride`` (and the channel change).
    """

    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        conv_bias: bool = _CONV_BIAS,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = [
            ConvNormAct(in_channels, out_channels, kernel_size, stride, conv_bias)
        ]
        for _ in range(num_convs - 1):
            blocks.append(
                ConvNormAct(out_channels, out_channels, kernel_size, (1, 1), conv_bias)
            )
        self.convs = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class EncoderStage(nn.Module):
    """One contracting-path stage (downsampling by a strided convolution)."""

    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        conv_bias: bool = _CONV_BIAS,
    ) -> None:
        super().__init__()
        self.block = StackedConvBlock(num_convs, in_channels, out_channels,
                                      kernel_size, stride, conv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    """One expanding-path stage.

    Upsamples the deeper feature map with a ``ConvTranspose2d`` (kernel = stride),
    concatenates the encoder skip, then applies a ``StackedConvBlock``.
    """

    def __init__(
        self,
        num_convs: int,
        in_channels_below: int,
        skip_channels: int,
        num_classes: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        conv_bias: bool = _CONV_BIAS,
    ) -> None:
        super().__init__()
        self.transpconv = nn.ConvTranspose2d(
            in_channels_below, skip_channels,
            kernel_size=stride, stride=stride, bias=conv_bias,
        )
        # After concatenation the channels are 2 * skip_channels.
        self.block = StackedConvBlock(num_convs, 2 * skip_channels, skip_channels,
                                      kernel_size, (1, 1), conv_bias)
        # Segmentation head of this resolution (unused in feature mode).
        self.seg_layer = nn.Conv2d(skip_channels, num_classes, kernel_size=1)

    def forward(self, x_below: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.transpconv(x_below)
        # Pad to the skip size if the input was not divisible by 2^depth
        # (a no-op for 752 x 256 inputs).
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class OutConv(nn.Module):
    """1x1 convolution producing segmentation logits (unused in feature mode)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet_nnunet(nn.Module):
    """nnU-Net v2 ``PlainConvUNet`` reproduction, in feature-extractor mode.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    n_features_map : int, default 512
        Channels of the returned full-resolution feature map: a final 1x1
        ``ConvNormAct`` head projects the last decoder features to that size.
    num_classes : int, default 2
        Channels of the (unused) segmentation heads; kept at the value the
        released checkpoints were trained with.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_features_map: int = 512,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = _FEATURES_PER_STAGE,
        kernel_sizes: Sequence[Tuple[int, int]] = _KERNEL_SIZES,
        strides: Sequence[Tuple[int, int]] = _STRIDES,
        n_conv_per_stage: Sequence[int] = _N_CONV_PER_STAGE,
        n_conv_per_stage_decoder: Sequence[int] = _N_CONV_PER_STAGE_DECODER,
        conv_bias: bool = _CONV_BIAS,
    ) -> None:
        super().__init__()
        self.n_stages = len(features_per_stage)
        assert len(kernel_sizes) == self.n_stages
        assert len(strides) == self.n_stages
        assert len(n_conv_per_stage) == self.n_stages
        assert len(n_conv_per_stage_decoder) == self.n_stages - 1

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features_per_stage = tuple(features_per_stage)
        self.strides = tuple(tuple(s) for s in strides)

        # ----- Encoder (7 stages; stage 0 keeps resolution, stages 1..6 /2) ---
        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for s in range(self.n_stages):
            self.encoder.append(
                EncoderStage(
                    num_convs=n_conv_per_stage[s],
                    in_channels=prev_channels,
                    out_channels=features_per_stage[s],
                    kernel_size=tuple(kernel_sizes[s]),
                    stride=tuple(strides[s]),
                    conv_bias=conv_bias,
                )
            )
            prev_channels = features_per_stage[s]

        # ----- Decoder (n_stages - 1 stages, mirroring the encoder) -----------
        self.decoder = nn.ModuleList()
        for d in range(self.n_stages - 1):
            s = d + 1                                     # from the bottleneck upward
            in_below = features_per_stage[-s]             # deeper stage channels
            skip_channels = features_per_stage[-(s + 1)]  # encoder skip channels
            stride = tuple(strides[-s])                   # upsampling factor
            self.decoder.append(
                DecoderStage(
                    num_convs=n_conv_per_stage_decoder[d],
                    in_channels_below=in_below,
                    skip_channels=skip_channels,
                    num_classes=num_classes,
                    kernel_size=tuple(kernel_sizes[-(s + 1)]),
                    stride=stride,
                    conv_bias=conv_bias,
                )
            )
        # Full-resolution segmentation head, tied to the last decoder seg_layer.
        self.output_head: OutConv = OutConv(features_per_stage[0], num_classes)
        self.decoder[-1].seg_layer = self.output_head.conv

        # ----- Feature head ---------------------------------------------------
        # 1x1 Conv -> InstanceNorm -> LeakyReLU projecting the last decoder
        # features (features_per_stage[0]) up to n_features_map channels.
        self.n_features_map = int(n_features_map)
        self.feature_head = ConvNormAct(
            features_per_stage[0], self.n_features_map,
            kernel_size=(1, 1), stride=(1, 1), conv_bias=conv_bias,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """nnU-Net's ``InitWeights_He(1e-2)``: He-normal conv weights, zero bias."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: keep every stage output for the skip connections.
        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)

        # Decoder: walk from the bottleneck upward, consuming skips in reverse.
        lres_input = skips[-1]
        for d, stage in enumerate(self.decoder):
            lres_input = stage(lres_input, skips[-(d + 2)])

        return self.feature_head(lres_input)  # (B, n_features_map, H, W)
