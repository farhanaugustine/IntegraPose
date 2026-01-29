from __future__ import annotations

import math
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


if __package__ is None or __package__ == "":  # pragma: no cover
    from config import MODEL_DEFAULTS
else:
    from .config import MODEL_DEFAULTS


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
        base_channels: int = MODEL_DEFAULTS["slowfast_base_channels"],
        alpha: int = MODEL_DEFAULTS["slowfast_alpha"],
        fusion_ratio: float = MODEL_DEFAULTS["slowfast_fusion_ratio"],
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


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, kind: str = "none", max_len: int = MODEL_DEFAULTS["positional_encoding_max_len"]):
        super().__init__()
        kind = str(kind or "none").strip().lower()
        if kind not in {"none", "sinusoidal", "learned"}:
            raise ValueError(
                "Unsupported positional encoding "
                f"{kind!r}. Use 'none', 'sinusoidal', or 'learned'."
            )
        self.kind = kind
        self.dim = int(dim)
        self.max_len = int(max_len)
        if self.kind == "learned":
            if self.max_len <= 0:
                raise ValueError("positional_encoding_max_len must be >= 1")
            self.embedding = nn.Embedding(self.max_len, self.dim)
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            self.embedding = None

    @staticmethod
    def _sinusoidal(length: int, dim: int, device: torch.device) -> torch.Tensor:
        if length <= 0:
            raise ValueError("length must be >= 1")
        if dim <= 0:
            raise ValueError("dim must be >= 1")
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / float(dim))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_cols = pe[:, 1::2].shape[1]
        if cos_cols:
            pe[:, 1::2] = torch.cos(position * div_term[:cos_cols])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return x
        if x.ndim != 3:
            raise ValueError("Expected input with shape [B, T, D].")
        _, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(
                f"Positional encoding dim mismatch: expected {self.dim}, got {dim}."
            )

        if self.kind == "learned":
            if length > self.max_len:
                raise ValueError(
                    f"Sequence length {length} exceeds learned positional encoding "
                    f"max_len={self.max_len}."
                )
            pos = torch.arange(length, device=x.device, dtype=torch.long)
            pe = self.embedding(pos).unsqueeze(0).to(dtype=x.dtype)
            return x + pe

        pe = self._sinusoidal(length, dim, device=x.device).to(dtype=x.dtype)
        return x + pe.unsqueeze(0)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = MODEL_DEFAULTS["attention_heads"],
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
    def __init__(self, dim: int, num_heads: int = MODEL_DEFAULTS["attention_heads"], dropout: float = 0.1):
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


class PretrainedSlowFast(nn.Module):
    """
    Wrapper for the official PyTorchVideo SlowFast R50 model (Kinetics-400).
    Input: [B, T, C, H, W]
    Output: [B, 2304] (Feature embedding)
    """
    def __init__(self, alpha: int = 4, pretrained: bool = True):
        super().__init__()
        self.alpha = alpha
        try:
            # We use the torch.hub interface
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                'slowfast_r50', 
                pretrained=pretrained
            )
        except ImportError:
            raise ImportError(
                "pytorchvideo is required for 'slowfast_r50'. "
                "Install it or check your internet connection for torch.hub."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SlowFast from torch.hub: {e}")

        # The last block (blocks[6]) is the head. We only want the backbone.
        self.backbone_blocks = nn.ModuleList(self.model.blocks[:-1])
        
        # We perform global average pooling manually to be robust
        self.pool = nn.AdaptiveAvgPool3d(1)
            
        self.output_dim = 2304

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] (Caller BehaviorSequenceClassifier already permuted it)
        
        # Prepare pathways
        # Fast: All frames
        # Slow: Stride alpha
        fast_path = x
        slow_path = x[:, :, ::self.alpha, :, :]
        
        inputs = [slow_path, fast_path]
        
        # Run backbone
        for block in self.backbone_blocks:
            inputs = block(inputs)
            
        if isinstance(inputs, torch.Tensor):
            out = self.pool(inputs)
        else:
            # Inputs is a list of feature maps [slow_feat, fast_feat]
            # Pool each pathway
            pooled_outputs = []
            for feat in inputs:
                pooled_outputs.append(self.pool(feat))
            
            # Concatenate: [B, C_slow, 1, 1, 1] + [B, C_fast, 1, 1, 1] -> [B, C_total, 1, 1, 1]
            out = torch.cat(pooled_outputs, dim=1)
        
        return out.flatten(1)


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
    elif name == "slowfast_r50":
        # External SlowFast (Kinetics-400)
        # Note: 'trainable' flag behavior:
        # If False, we freeze the whole thing. If True, we fine-tune.
        encoder = PretrainedSlowFast(alpha=slowfast_kwargs.get('alpha', 4), pretrained=pretrained)
        feature_dim = encoder.output_dim
        if not trainable:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder, feature_dim
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose from ['mobilenet_v3_small', 'efficientnet_b0', 'vit_b_16', 'slowfast_tiny', 'slowfast_r50']."
        )

    if not trainable:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder, feature_dim


class PoseVisualFusion(nn.Module):
    """
    Module to fuse visual features and pose features.
    Supports simple concatenation, gated attention, or a small cross-modal transformer.

    Note: Fusion operates on feature vectors (channel/token space), not a spatial mask over pixels.
    """
    def __init__(
        self,
        visual_dim: int,
        pose_dim: int,
        fusion_dim: int = 128,
        strategy: str = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.strategy = strategy.lower()
        self.visual_dim = visual_dim
        self.pose_dim = pose_dim
        self.fusion_dim = fusion_dim

        # Project pose to a higher dimensional embedding
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.strategy == "concat":
            self.output_dim = visual_dim + fusion_dim
        
        elif self.strategy == "gated_attention":
            # "Sweet Spot": Use Pose+Visual to gate the Visual features.
            # 1. Concatenate Visual and Projected Pose
            # 2. Predict channel-wise attention weights for Visual
            # 3. Output = Concat(WeightedVisual, ProjectedPose)
            
            # Bottleneck for the gate calculation
            reduction = 4
            gate_input_dim = visual_dim + fusion_dim
            hidden_dim = max(gate_input_dim // reduction, 16)
            
            self.gate_fc = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, visual_dim),
                nn.Sigmoid(),
            )
            self.output_dim = visual_dim + fusion_dim
        
        elif self.strategy == "cross_modal_transformer":
            # "Deep Integration": Treat Visual and Pose as two tokens in a sequence and use Self-Attention.
            # This allows bidirectional flow: Visual context refines Pose, Pose context refines Visual.
            
            # We project Pose to match Visual dimension to stack them
            self.pose_to_visual = nn.Linear(fusion_dim, visual_dim)
            
            # A small transformer block to mix the two modalities
            # d_model = visual_dim
            self.modal_mixer = nn.TransformerEncoderLayer(
                d_model=visual_dim,
                nhead=4,
                dim_feedforward=visual_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            
            # We will concatenate the two transformed tokens
            self.output_dim = visual_dim * 2

        else:
            raise ValueError(f"Unsupported pose_fusion_strategy '{strategy}'. Use 'concat', 'gated_attention', or 'cross_modal_transformer'.")

    def forward(
        self,
        visual_x: torch.Tensor,
        pose_x: torch.Tensor | None,
        pose_present: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_x: [B, T, Dv] or [B, Dv]
            pose_x:   [B, T, Dp] or [B, Dp] or None
            pose_present: optional boolean indicating if pose should be used.
                - bool or scalar: applied to all samples
                - [B] or [B, T]: per-sample/per-timestep
        """
        pose_missing = pose_x is None or (isinstance(pose_present, bool) and not pose_present)
        pose_mask = None
        if not pose_missing and isinstance(pose_present, torch.Tensor):
            pose_mask = pose_present.to(dtype=torch.bool, device=visual_x.device)

        # Upsample/Embed pose (or create a safe "missing pose" embedding)
        if pose_missing:
            if visual_x.ndim == 3:
                b, t, _ = visual_x.shape
                pose_emb = torch.zeros(
                    b, t, self.fusion_dim, device=visual_x.device, dtype=visual_x.dtype
                )
            else:
                b = visual_x.shape[0]
                pose_emb = torch.zeros(
                    b, self.fusion_dim, device=visual_x.device, dtype=visual_x.dtype
                )
        else:
            pose_emb = self.pose_encoder(pose_x)
            if pose_mask is not None:
                if visual_x.ndim == 3 and pose_mask.ndim == 1:
                    pose_emb = pose_emb * pose_mask[:, None, None]
                elif visual_x.ndim == 2 and pose_mask.ndim == 1:
                    pose_emb = pose_emb * pose_mask[:, None]
                elif visual_x.ndim == 3 and pose_mask.ndim == 2:
                    pose_emb = pose_emb * pose_mask[:, :, None]
                else:
                    raise ValueError(
                        f"Unsupported pose_present shape {pose_mask.shape} for visual_x shape {visual_x.shape}."
                    )
        
        if self.strategy == "concat":
            return torch.cat([visual_x, pose_emb], dim=-1)
            
        elif self.strategy == "gated_attention":
            combined = torch.cat([visual_x, pose_emb], dim=-1)
            gates = None
            if not pose_missing:
                gates = self.gate_fc(combined)  # [B, T, Dv] or [B, Dv]

            if pose_missing:
                visual_weighted = visual_x
            elif pose_mask is None:
                visual_weighted = visual_x * gates
            else:
                # Compute gating for the batch, but force RGB-only behavior where pose is missing.
                if visual_x.ndim == 3 and pose_mask.ndim == 1:
                    gate_mask = pose_mask[:, None, None]
                elif visual_x.ndim == 2 and pose_mask.ndim == 1:
                    gate_mask = pose_mask[:, None]
                elif visual_x.ndim == 3 and pose_mask.ndim == 2:
                    gate_mask = pose_mask[:, :, None]
                else:
                    raise ValueError(
                        f"Unsupported pose_present shape {pose_mask.shape} for visual_x shape {visual_x.shape}."
                    )
                gates = gates * gate_mask + (1.0 - gate_mask.to(dtype=gates.dtype))  # missing -> 1s
                visual_weighted = visual_x * gates
            
            # Fuse
            return torch.cat([visual_weighted, pose_emb], dim=-1)
            
        elif self.strategy == "cross_modal_transformer":
            bypass = torch.cat([visual_x, torch.zeros_like(visual_x)], dim=-1)
            if pose_missing:
                return bypass
            # 1. Project pose to visual dim
            pose_proj = self.pose_to_visual(pose_emb) # [B, T, Dv] or [B, Dv]
            
            # Check dimensions
            is_sequence = (visual_x.ndim == 3) # [B, T, Dv]
            
            if is_sequence:
                b, t, d = visual_x.shape
                # Reshape to [B*T, 1, D] to process each timestep as an independent sample
                v_token = visual_x.view(b * t, 1, d)
                p_token = pose_proj.view(b * t, 1, d)
                
                # [B*T, 2, D]
                tokens = torch.cat([v_token, p_token], dim=1)
                
                # Apply Transformer
                transformed = self.modal_mixer(tokens) # [B*T, 2, D]
                
                # Flatten back to [B, T, 2*D]
                out = transformed.view(b, t, d * 2)
                if pose_mask is None:
                    return out
                if pose_mask.ndim != 1:
                    raise ValueError(
                        f"Unsupported pose_present shape {pose_mask.shape} for cross_modal_transformer (sequence)."
                    )
                m = pose_mask[:, None, None].to(dtype=out.dtype)
                return out * m + bypass * (1.0 - m)
            else:
                # [B, Dv] case (SlowFast / Global features)
                b, d = visual_x.shape
                
                # Treat as sequence length 1 for transformer logic? 
                # Or just stack as [B, 2, D] directly.
                v_token = visual_x.unsqueeze(1) # [B, 1, D]
                p_token = pose_proj.unsqueeze(1) # [B, 1, D]
                
                tokens = torch.cat([v_token, p_token], dim=1) # [B, 2, D]
                
                transformed = self.modal_mixer(tokens) # [B, 2, D]
                
                # Flatten to [B, 2*D]
                out = transformed.view(b, d * 2)
                if pose_mask is None:
                    return out
                if pose_mask.ndim != 1:
                    raise ValueError(
                        f"Unsupported pose_present shape {pose_mask.shape} for cross_modal_transformer (global)."
                    )
                m = pose_mask[:, None].to(dtype=out.dtype)
                return out * m + bypass * (1.0 - m)
        
        return torch.cat([visual_x, pose_emb], dim=-1)


class BehaviorSequenceClassifier(nn.Module):
    """
    Flexible behavior classifier supporting LSTM, attention pooling, and SlowFast backbone.
    Supports optional pose feature fusion.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenet_v3_small",
        pretrained_backbone: bool = True,
        train_backbone: bool = False,
        hidden_dim: int = MODEL_DEFAULTS["hidden_dim"],
        num_layers: int = MODEL_DEFAULTS["num_layers"],
        dropout: float = MODEL_DEFAULTS["dropout"],
        bidirectional: bool = False,
        sequence_model: str = "lstm",
        temporal_attention_layers: int = 0,
        attention_heads: int = MODEL_DEFAULTS["attention_heads"],
        positional_encoding: str = "none",
        positional_encoding_max_len: int = MODEL_DEFAULTS["positional_encoding_max_len"],
        use_attention_pooling: bool = False,
        use_feature_se: bool = False,
        slowfast_alpha: int = MODEL_DEFAULTS["slowfast_alpha"],
        slowfast_fusion_ratio: float = MODEL_DEFAULTS["slowfast_fusion_ratio"],
        slowfast_base_channels: int = MODEL_DEFAULTS["slowfast_base_channels"],
        pose_input_dim: int = 0,
        pose_fusion_dim: int = 128,
        pose_fusion_strategy: str = "gated_attention",
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.sequence_model_type = sequence_model
        self.use_attention_pooling = use_attention_pooling
        self.pose_input_dim = pose_input_dim
        self.pose_fusion_dim = pose_fusion_dim
        self.pose_fusion_strategy = pose_fusion_strategy

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
            "video" if self.backbone_name.startswith("slowfast") else "frame"
        )
        
        # Pose Fusion Module
        self.pose_fusion = None
        if pose_input_dim > 0:
            self.pose_fusion = PoseVisualFusion(
                visual_dim=feature_dim,
                pose_dim=pose_input_dim,
                fusion_dim=pose_fusion_dim,
                strategy=pose_fusion_strategy,
                dropout=dropout
            )
            feature_dim = self.pose_fusion.output_dim

        positional_encoding = str(positional_encoding or "none").strip().lower()
        self.frame_positional_encoding = (
            TemporalPositionalEncoding(
                feature_dim,
                kind=positional_encoding,
                max_len=positional_encoding_max_len,
            )
            if self.backbone_type == "frame"
            and positional_encoding != "none"
            and (
                temporal_attention_layers > 0
                or str(sequence_model).lower() == "attention"
            )
            else None
        )

        self.pool_positional_encoding: TemporalPositionalEncoding | None = None

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

            if (
                positional_encoding != "none"
                and sequence_model == "lstm"
                and self.use_attention_pooling
            ):
                self.pool_positional_encoding = TemporalPositionalEncoding(
                    seq_output_dim,
                    kind=positional_encoding,
                    max_len=positional_encoding_max_len,
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
            # For SlowFast, feature_dim has already been updated to include pose fusion dim
            classifier_input = feature_dim

        self.head = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(dropout),
            nn.Linear(classifier_input, num_classes),
        )

    def forward(
        self,
        inputs: torch.Tensor
        | Tuple[torch.Tensor, torch.Tensor | None]
        | Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    ) -> torch.Tensor:
        """
        Args:
            inputs:
                - clip: Tensor [B, T, C, H, W]
                - optional pose: Tensor [B, T, P] | None
                - optional pose_present: Tensor [B] or [B, T] | None
        """
        clip = inputs
        pose = None
        pose_present = None
        if isinstance(inputs, tuple):
            if len(inputs) == 2:
                clip, pose = inputs
            elif len(inputs) == 3:
                clip, pose, pose_present = inputs
            else:
                raise ValueError("Expected inputs tuple of (clip, pose) or (clip, pose, pose_present).")
        
        if clip.ndim != 5:
            raise ValueError("Expected input tensor with shape [B, T, C, H, W]")

        if self.backbone_type == "video":
            video_clip = clip.permute(0, 2, 1, 3, 4).contiguous()
            features = self.frame_encoder(video_clip)
            
            # Late fusion for SlowFast
            if self.pose_fusion is not None:
                pose_flat = None
                if pose is not None:
                    if pose.ndim == 4:
                        b, t, *_ = pose.shape
                        pose_flat = pose.view(b, t, -1)
                    elif pose.ndim == 3:
                        pose_flat = pose
                    elif pose.ndim == 2:
                        pose_flat = pose.unsqueeze(1)
                    else:
                        raise ValueError(f"Unsupported pose tensor shape for SlowFast: {pose.shape}")

                pooled_present = None
                if pose_present is not None:
                    if pose_present.ndim == 2:
                        pooled_present = pose_present.any(dim=1)
                    else:
                        pooled_present = pose_present

                if pose_flat is not None and pose_flat.ndim == 3:
                    pose_pooled = pose_flat.mean(dim=1)  # [B, P]
                elif pose_flat is not None and pose_flat.ndim == 2:
                    pose_pooled = pose_flat
                else:
                    pose_pooled = None

                features = self.pose_fusion(features, pose_pooled, pose_present=pooled_present)
                
            logits = self.head(features)
            return logits

        batch_size, num_frames, _, _, _ = clip.shape
        flattened = clip.view(batch_size * num_frames, *clip.shape[2:])
        frame_features = self.frame_encoder(flattened)
        frame_features = frame_features.view(batch_size, num_frames, -1)  
        
        # Pose Fusion (Frame-level)
        if self.pose_fusion is not None:
            pose_flat = None
            if pose is not None:
                if pose.ndim == 4:
                    pose_flat = pose.view(batch_size, num_frames, -1)
                elif pose.ndim == 3:
                    pose_flat = pose
                else:
                    raise ValueError(f"Unsupported pose tensor shape: {pose.shape}")

            frame_features = self.pose_fusion(frame_features, pose_flat, pose_present=pose_present)

        if self.feature_se is not None:
            frame_features = self.feature_se(frame_features)
        if self.frame_positional_encoding is not None:
            frame_features = self.frame_positional_encoding(frame_features)
        if self.temporal_blocks is not None:
            for block in self.temporal_blocks:
                frame_features = block(frame_features)

        if self.sequence_model_type == "lstm":
            sequence_out, _ = self.sequence_model(frame_features)
            if self.attention_pool is not None:
                if self.pool_positional_encoding is not None:
                    sequence_out = self.pool_positional_encoding(sequence_out)
                pooled = self.attention_pool(sequence_out)
            else:
                pooled = sequence_out[:, -1, :]
        else:
            pooled = self.attention_pool(frame_features)

        logits = self.head(pooled)
        return logits
