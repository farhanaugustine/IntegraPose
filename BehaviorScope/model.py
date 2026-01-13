from __future__ import annotations

import warnings
from typing import Tuple

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

try:  # pragma: no cover - optional dependency (torchvision version)
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except Exception:  # pragma: no cover
    ViT_B_16_Weights = None  # type: ignore[assignment]
    vit_b_16 = None  # type: ignore[assignment]


class Conv3dBNRelu(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] | None = None,
    ):
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class Residual3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        self.conv1 = Conv3dBNRelu(in_channels, out_channels, stride=stride)
        self.conv2 = Conv3dBNRelu(out_channels, out_channels, stride=(1, 1, 1))
        if in_channels != out_channels or stride != (1, 1, 1):
            self.proj = Conv3dBNRelu(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                stride=stride,
                padding=(0, 0, 0),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.relu(x + residual)


class SlowFastBackbone(nn.Module):
    """
    Lightweight SlowFast-style backbone that keeps separate slow/fast pathways.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        alpha: int = 4,
        fusion_ratio: float = 0.25,
    ):
        super().__init__()
        if alpha < 1:
            raise ValueError("alpha must be >= 1")
        self.alpha = alpha
        slow_channels = int(base_channels * fusion_ratio)
        slow_channels = max(slow_channels, 16)
        fast_channels = base_channels

        self.fast_path = nn.Sequential(
            Conv3dBNRelu(
                in_channels,
                fast_channels,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
            ),
            Residual3DBlock(
                fast_channels, fast_channels * 2, stride=(1, 2, 2)
            ),
            Residual3DBlock(fast_channels * 2, fast_channels * 2),
        )
        slow_input = max(1, in_channels)
        self.slow_path = nn.Sequential(
            Conv3dBNRelu(
                slow_input,
                slow_channels,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
            ),
            Residual3DBlock(
                slow_channels, slow_channels * 2, stride=(1, 2, 2)
            ),
            Residual3DBlock(slow_channels * 2, slow_channels * 2),
        )
        self.lateral = nn.Sequential(
            nn.AvgPool3d(kernel_size=(alpha, 1, 1), stride=(alpha, 1, 1)),
            Conv3dBNRelu(
                fast_channels * 2, slow_channels * 2, kernel_size=(3, 1, 1)
            ),
        )
        fused_channels = slow_channels * 2 + slow_channels * 2
        self.fusion = Residual3DBlock(fused_channels, slow_channels * 4)

        self.avgpool_slow = nn.AdaptiveAvgPool3d(1)
        self.avgpool_fast = nn.AdaptiveAvgPool3d(1)
        self.output_dim = slow_channels * 4 + fast_channels * 2

    def _temporal_subsample(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :: self.alpha, :, :]

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        fast = self.fast_path(clip)
        slow = self.slow_path(self._temporal_subsample(clip))
        lateral = self.lateral(fast)
        if lateral.shape[2] != slow.shape[2]:
            lateral = torch.nn.functional.interpolate(
                lateral,
                size=slow.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        fused = torch.cat([slow, lateral], dim=1)
        slow = self.fusion(fused)

        slow_feat = self.avgpool_slow(slow).flatten(1)
        fast_feat = self.avgpool_fast(fast).flatten(1)
        return torch.cat([slow_feat, fast_feat], dim=1)


class FeatureSELayer(nn.Module):
    def __init__(self, feature_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(feature_dim // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x.mean(dim=1))
        return x * scale.unsqueeze(1)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        query = self.query.expand(b, -1, -1)
        attn_out, _ = self.attn(query, x, x)
        pooled = self.norm(attn_out.squeeze(1))
        return pooled


def build_frame_encoder(
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
    trainable: bool = False,
    slowfast_kwargs: dict | None = None,
) -> Tuple[nn.Module, int]:
    """
    Create a lightweight CNN feature extractor that converts each frame to a
    compact embedding. Returns the module and the embedding dimensionality.
    """
    name = backbone.lower()

    def _instantiate(builder, weights):
        try:
            return builder(weights=weights)
        except Exception as exc:
            if weights is not None:
                warnings.warn(
                    f"Falling back to randomly initialized weights for {backbone}: {exc}"
                )
                return builder(weights=None)
            raise

    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = _instantiate(models.mobilenet_v3_small, weights)
        feature_dim = model.classifier[0].in_features
        encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = _instantiate(efficientnet_b0, weights)
        feature_dim = model.classifier[1].in_features
        encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    elif name == "vit_b_16":
        if vit_b_16 is None:
            raise ValueError(
                "Unsupported backbone 'vit_b_16' (VisionTransformer not available in this torchvision)."
            )
        weights = (
            ViT_B_16_Weights.DEFAULT
            if pretrained and ViT_B_16_Weights is not None
            else None
        )
        model = _instantiate(vit_b_16, weights)
        feature_dim = int(getattr(model, "hidden_dim", 768))
        model.heads = nn.Identity()
        encoder = model
    elif name == "slowfast_tiny":
        encoder = SlowFastBackbone(**(slowfast_kwargs or {}))
        feature_dim = encoder.output_dim
        if not trainable:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder, feature_dim
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose from ['mobilenet_v3_small', 'efficientnet_b0', 'vit_b_16', 'slowfast_tiny']."
        )

    if not trainable:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder, feature_dim


class BehaviorSequenceClassifier(nn.Module):
    """
    Flexible behavior classifier supporting LSTM, attention pooling, and SlowFast backbone.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenet_v3_small",
        pretrained_backbone: bool = True,
        train_backbone: bool = False,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        sequence_model: str = "lstm",
        temporal_attention_layers: int = 0,
        attention_heads: int = 4,
        use_attention_pooling: bool = False,
        use_feature_se: bool = False,
        slowfast_alpha: int = 4,
        slowfast_fusion_ratio: float = 0.25,
        slowfast_base_channels: int = 48,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.sequence_model_type = sequence_model
        self.use_attention_pooling = use_attention_pooling

        self.frame_encoder, feature_dim = build_frame_encoder(
            backbone=backbone,
            pretrained=pretrained_backbone,
            trainable=train_backbone,
            slowfast_kwargs={
                "alpha": slowfast_alpha,
                "fusion_ratio": slowfast_fusion_ratio,
                "base_channels": slowfast_base_channels,
            },
        )
        self.backbone_type = (
            "video" if self.backbone_name == "slowfast_tiny" else "frame"
        )

        self.feature_se = (
            FeatureSELayer(feature_dim) if use_feature_se else None
        )
        if temporal_attention_layers > 0 and self.backbone_type == "frame":
            self.temporal_blocks = nn.ModuleList(
                [
                    TemporalSelfAttentionBlock(
                        feature_dim, num_heads=attention_heads, dropout=dropout
                    )
                    for _ in range(temporal_attention_layers)
                ]
            )
        else:
            self.temporal_blocks = None

        if self.backbone_type == "frame":
            if sequence_model == "lstm":
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.sequence_model = nn.LSTM(
                    input_size=feature_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                seq_output_dim = hidden_dim * (2 if bidirectional else 1)
            elif sequence_model == "attention":
                self.sequence_model = None
                seq_output_dim = feature_dim
                self.use_attention_pooling = True
            else:
                raise ValueError(
                    f"Unsupported sequence_model '{sequence_model}'. "
                    "Use 'lstm' or 'attention'."
                )

            if self.use_attention_pooling:
                self.attention_pool = TemporalAttentionPooling(
                    seq_output_dim, num_heads=attention_heads, dropout=dropout
                )
            else:
                self.attention_pool = None

            classifier_input = seq_output_dim
        else:
            self.sequence_model = None
            self.attention_pool = None
            classifier_input = feature_dim

        self.head = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(dropout),
            nn.Linear(classifier_input, num_classes),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: Tensor with shape [batch, frames, channels, height, width]
        """
        if clip.ndim != 5:
            raise ValueError("Expected input tensor with shape [B, T, C, H, W]")

        if self.backbone_type == "video":
            video_clip = clip.permute(0, 2, 1, 3, 4).contiguous()
            features = self.frame_encoder(video_clip)
            logits = self.head(features)
            return logits

        batch_size, num_frames, _, _, _ = clip.shape
        flattened = clip.view(batch_size * num_frames, *clip.shape[2:])
        frame_features = self.frame_encoder(flattened)
        frame_features = frame_features.view(batch_size, num_frames, -1)

        if self.feature_se is not None:
            frame_features = self.feature_se(frame_features)
        if self.temporal_blocks is not None:
            for block in self.temporal_blocks:
                frame_features = block(frame_features)

        if self.sequence_model_type == "lstm":
            sequence_out, _ = self.sequence_model(frame_features)
            if self.attention_pool is not None:
                pooled = self.attention_pool(sequence_out)
            else:
                pooled = sequence_out[:, -1, :]
        else:
            pooled = self.attention_pool(frame_features)

        logits = self.head(pooled)
        return logits
