"""TandemYTC pose/social temporal classifier."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from config import MODEL_DEFAULTS
except Exception:  # pragma: no cover
    from .config import MODEL_DEFAULTS


def inspect_yolo_keypoint_count(weights_path: str) -> Optional[int]:
    """Return the number of keypoints emitted by a YOLO-pose checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    try:
        yolo = YOLO(weights_path)
    except Exception:
        return None
    if getattr(yolo, "task", None) != "pose":
        return None
    kpt_shape = getattr(yolo.model, "kpt_shape", None)
    if kpt_shape is None:
        head = yolo.model.model[-1]
        kpt_shape = getattr(head, "kpt_shape", None)
    if kpt_shape is None:
        return None
    try:
        return int(kpt_shape[0])
    except (TypeError, IndexError):
        return None


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.to(x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    summed = (x * m).sum(dim=dim)
    denom = m.sum(dim=dim).clamp(min=1.0)
    return summed / denom


class TemporalPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        kind: str = "none",
        max_len: int = MODEL_DEFAULTS["positional_encoding_max_len"],
    ) -> None:
        super().__init__()
        kind = str(kind or "none").strip().lower()
        if kind not in {"none", "sinusoidal", "learned"}:
            raise ValueError(f"Unsupported positional encoding {kind!r}.")
        self.kind = kind
        self.dim = int(dim)
        self.max_len = int(max_len)
        if self.kind == "learned":
            self.embedding = nn.Embedding(self.max_len, self.dim)
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            self.embedding = None

    @staticmethod
    def _sinusoidal(length: int, dim: int, device: torch.device) -> torch.Tensor:
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
            raise ValueError(f"Positional dim mismatch: expected {self.dim}, got {dim}.")
        if self.kind == "learned":
            if length > self.max_len:
                raise ValueError(f"Sequence length {length} exceeds max_len={self.max_len}.")
            pos = torch.arange(length, device=x.device, dtype=torch.long)
            return x + self.embedding(pos).unsqueeze(0).to(dtype=x.dtype)
        return x + self._sinusoidal(length, dim, x.device).to(dtype=x.dtype).unsqueeze(0)


class TemporalAttentionPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = MODEL_DEFAULTS["attention_heads"],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(x.shape[0], -1, -1)
        attn_out, _ = self.attn(query, x, x)
        return self.norm(attn_out.squeeze(1))


class PoseSelfEncoder(nn.Module):
    def __init__(self, pose_input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalTemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.left_pad = int(dilation) * (int(kernel_size) - 1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, 1),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        z = F.pad(x.transpose(1, 2), (self.left_pad, 0))
        z = self.conv(z).transpose(1, 2)
        return self.norm(torch.relu(z + residual))


class TemporalConvHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        current_dim = int(in_dim)
        for idx in range(max(1, int(num_layers))):
            layers.append(
                CausalTemporalConvBlock(
                    current_dim,
                    int(hidden_dim),
                    dilation=2 ** idx,
                    dropout=dropout,
                )
            )
            current_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedTemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        weights = torch.softmax(self.score(z).squeeze(-1), dim=1)
        gated = x * self.gate(z)
        return self.out_norm(torch.sum(gated * weights.unsqueeze(-1), dim=1))


class MultiAnimalBehaviorSequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_animals: int = 2,
        num_keypoints: int = 7,
        rel_feature_dim: int = 11,
        hidden_dim: int = MODEL_DEFAULTS["hidden_dim"],
        dropout: float = MODEL_DEFAULTS["dropout"],
        sequence_model: str = "lstm",
        num_lstm_layers: int = MODEL_DEFAULTS["num_layers"],
        bidirectional_lstm: bool = False,
        use_attention_pool: bool = False,
        attention_heads: int = MODEL_DEFAULTS["attention_heads"],
        positional_encoding: str = "none",
        positional_encoding_max_len: int = MODEL_DEFAULTS["positional_encoding_max_len"],
        disable_pose_self: bool = False,
        disable_relations: bool = False,
        relations_pose_only: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        self.n_animals = int(n_animals)
        self.num_keypoints = int(num_keypoints)
        self.rel_feature_dim = int(rel_feature_dim)
        self.disable_pose_self = bool(disable_pose_self)
        self.disable_relations = bool(disable_relations)
        self.relations_pose_only = bool(relations_pose_only)
        self.pose_self_alive = not self.disable_pose_self
        self.feature_source_name = "yolo_pose"
        self.feature_source_type = "pose_social"

        d_pose_self = num_keypoints * 4 + num_keypoints * (num_keypoints - 1) // 2
        d_pose_input = d_pose_self + 4
        self.pose_self_encoder = PoseSelfEncoder(d_pose_input, hidden_dim, dropout)
        self.fused_token_dim = int(hidden_dim)

        d_rel_input = rel_feature_dim + 2
        self.relation_encoder = RelationEncoder(d_rel_input, hidden_dim, dropout)
        self.rel_token_dim = self.relation_encoder.out_dim

        combined_dim = 0
        if self.pose_self_alive:
            combined_dim += self.fused_token_dim
        if not self.disable_relations:
            combined_dim += self.rel_token_dim
        if combined_dim == 0:
            raise ValueError("At least one of pose-self or relations must be enabled.")
        self.combined_dim = int(combined_dim)

        self.positional_encoding = (
            TemporalPositionalEncoding(
                self.combined_dim,
                kind=str(positional_encoding or "none").strip().lower(),
                max_len=int(positional_encoding_max_len),
            )
            if positional_encoding and positional_encoding != "none"
            else None
        )

        aliases = {
            "attention": "attention",
            "lstm_attention": "attention_lstm",
            "attention_lstm": "attention_lstm",
            "gated_attention_lstm": "gated_attention_lstm",
            "lstm": "lstm",
            "tcn": "tcn",
            "tcn_attention": "tcn_attention",
        }
        self.sequence_model_type = aliases.get(str(sequence_model or "lstm").strip().lower(), str(sequence_model))
        self.use_attention_pool = bool(use_attention_pool)

        if self.sequence_model_type in {"lstm", "attention_lstm", "gated_attention_lstm"}:
            self.temporal_head = nn.LSTM(
                input_size=self.combined_dim,
                hidden_size=hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=bidirectional_lstm,
                dropout=dropout if num_lstm_layers > 1 else 0.0,
            )
            head_in_dim = hidden_dim * (2 if bidirectional_lstm else 1)
            if self.sequence_model_type == "gated_attention_lstm":
                self.attn_pool = GatedTemporalAttentionPooling(head_in_dim, dropout=dropout)
            elif self.sequence_model_type == "attention_lstm" or self.use_attention_pool:
                self.attn_pool = TemporalAttentionPooling(head_in_dim, int(attention_heads), dropout=dropout)
            else:
                self.attn_pool = None
        elif self.sequence_model_type in {"tcn", "tcn_attention"}:
            self.temporal_head = TemporalConvHead(
                in_dim=self.combined_dim,
                hidden_dim=hidden_dim,
                num_layers=num_lstm_layers,
                dropout=dropout,
            )
            head_in_dim = self.temporal_head.out_dim
            self.attn_pool = (
                TemporalAttentionPooling(head_in_dim, int(attention_heads), dropout=dropout)
                if self.sequence_model_type == "tcn_attention"
                else None
            )
        elif self.sequence_model_type == "attention":
            self.temporal_head = None
            self.attn_pool = TemporalAttentionPooling(
                self.combined_dim,
                int(attention_heads),
                dropout=dropout,
            )
            head_in_dim = self.combined_dim
        else:
            raise ValueError(f"Unsupported sequence_model: {sequence_model}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        nb = device.type == "cuda"
        animal_mask = batch["animal_mask"].to(device, non_blocking=nb)
        pose_mask = batch["pose_mask"].to(device, non_blocking=nb)
        pose_conf = batch["pose_conf"].to(device, non_blocking=nb)
        crop_conf = batch["crop_conf"].to(device, non_blocking=nb)
        track_conf = batch["track_conf"].to(device, non_blocking=nb)
        track_age = batch["track_age"].to(device, non_blocking=nb)
        pose_self = batch["pose_self"].to(device, non_blocking=nb)
        rel_features = batch["relation_features"].to(device, non_blocking=nb)
        rel_pose_conf = batch["relation_pose_conf"].to(device, non_blocking=nb)
        rel_present = batch["relation_present"].to(device, non_blocking=nb)

        b, t, n = animal_mask.shape
        parts = []
        if self.pose_self_alive:
            reliability = torch.stack([pose_conf, crop_conf, track_conf, track_age], dim=-1)
            pose_input = torch.cat([pose_self, reliability], dim=-1)
            if self.disable_pose_self:
                pose_input = torch.zeros_like(pose_input)
            pose_feat = self.pose_self_encoder(pose_input)
            parts.append(masked_mean(pose_feat, animal_mask, dim=2))

        if not self.disable_relations:
            rel_pose_conf_unsq = rel_pose_conf.unsqueeze(-1)
            rel_present_unsq = rel_present.float().unsqueeze(-1)
            rel_input = torch.cat([rel_features, rel_pose_conf_unsq, rel_present_unsq], dim=-1)
            if self.relations_pose_only:
                rel_input = rel_input.clone()
                rel_input[..., :6] = 0
            rel_tokens = self.relation_encoder(rel_input)
            rel_flat = rel_tokens.reshape(b, t, n * n, self.rel_token_dim)
            mask_flat = rel_present.reshape(b, t, n * n)
            parts.append(masked_mean(rel_flat, mask_flat, dim=2))

        combined = torch.cat(parts, dim=-1)
        if self.positional_encoding is not None:
            combined = self.positional_encoding(combined)

        if self.sequence_model_type in {"lstm", "attention_lstm", "gated_attention_lstm"}:
            seq_out, _ = self.temporal_head(combined)
            pooled = self.attn_pool(seq_out) if self.attn_pool is not None else seq_out[:, -1]
        elif self.sequence_model_type in {"tcn", "tcn_attention"}:
            seq_out = self.temporal_head(combined)
            pooled = self.attn_pool(seq_out) if self.attn_pool is not None else seq_out[:, -1]
        else:
            pooled = self.attn_pool(combined)
        return self.classifier(pooled)
