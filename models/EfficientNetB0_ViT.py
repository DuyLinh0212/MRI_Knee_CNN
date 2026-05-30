"""
EfficientNetB0_ViT v2 — Improved Architecture
================================================
Các cải tiến so với v1:
  1. SliceViT dùng nhiều Transformer layer (num_layers tham số hóa được).
  2. Thêm Stochastic Depth (DropPath) trong TransformerEncoderLayer.
  3. Fusion 3 plane bằng Cross-Attention thay vì đơn giản concat.
  4. Attention pooling thay vì max-pooling thuần túy.
  5. Positional encoding học được + fallback interpolation khi slices > max_slices.
  6. Head sâu hơn với BatchNorm và residual.
  7. Gradient checkpointing (tùy chọn) để tiết kiệm VRAM khi fine-tune backbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    _HAS_GRAD_CKPT = True
except ImportError:
    _HAS_GRAD_CKPT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_efficientnet_b0_features():
    try:
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        net = models.efficientnet_b0(pretrained=True)
    return net.features


class DropPath(nn.Module):
    """Stochastic depth: zeroes entire sample with probability `drop_prob`."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep).div_(keep)
        return x * mask


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Học một vector query để pool tập slice features thành đặc trưng duy nhất.
    Tốt hơn max-pool vì tập trung vào các slice quan trọng nhất.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        q = self.query.expand(x.size(0), -1, -1)          # [B, 1, D]
        out, _ = self.attn(q, x, x)                        # [B, 1, D]
        return out.squeeze(1)                               # [B, D]


# ---------------------------------------------------------------------------
# Improved SliceViT
# ---------------------------------------------------------------------------

class SliceViT(nn.Module):
    """
    Multi-layer Transformer + Stochastic Depth + Attention Pooling.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,          # v1 chỉ có 1 layer
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,  # stochastic depth
        max_slices: int = 64,
    ):
        super().__init__()
        self.max_slices = max_slices
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embed với kích thước linh hoạt (max_slices + 1 cho CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices + 1, embed_dim))

        # Stochastic depth rates tăng dần theo layer
        dpr = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]

        self.layers = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, dropout=dropout, drop_path=dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _interpolate_pos_embed(self, n: int) -> torch.Tensor:
        """Nội suy pos_embed khi số slice vượt max_slices."""
        pe = self.pos_embed[:, 1:, :]              # [1, max_slices, D]
        pe = pe.permute(0, 2, 1)                   # [1, D, max_slices]
        pe = F.interpolate(pe, size=n, mode="linear", align_corners=False)
        pe = pe.permute(0, 2, 1)                   # [1, n, D]
        cls_pe = self.pos_embed[:, :1, :]          # [1, 1, D]
        return torch.cat([cls_pe, pe], dim=1)      # [1, n+1, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, in_dim]
        B, S, _ = x.shape
        x = self.proj(x)                           # [B, S, embed_dim]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # [B, S+1, embed_dim]

        if S <= self.max_slices:
            x = x + self.pos_embed[:, : S + 1, :]
        else:
            # FIX v1: thay vì raise error, nội suy pos_embed
            x = x + self._interpolate_pos_embed(S).to(x.dtype)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0]                             # CLS token


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block với DropPath."""

    def __init__(self, dim: int, num_heads: int, dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop_path(h)
        h = self.ff(self.norm2(x))
        x = x + self.drop_path(h)
        return x


# ---------------------------------------------------------------------------
# Cross-Plane Fusion via Cross-Attention
# ---------------------------------------------------------------------------

class PlaneCrossFusion(nn.Module):
    """
    Fuse axial/coronal/sagittal features bằng cross-attention thay vì concat đơn giản.
    Mỗi plane attend vào 2 plane còn lại; output là trung bình cộng 3 refined features.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ca_axial = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ca_coronal = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ca_sagittal = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, axial, coronal, sagittal):
        # Mỗi tensor: [B, D]  →  unsqueeze thành [B, 1, D]
        a = axial.unsqueeze(1)
        c = coronal.unsqueeze(1)
        s = sagittal.unsqueeze(1)
        context = torch.cat([a, c, s], dim=1)  # [B, 3, D]

        a_out, _ = self.ca_axial(a, context, context)
        c_out, _ = self.ca_coronal(c, context, context)
        s_out, _ = self.ca_sagittal(s, context, context)

        fused = torch.cat([
            self.norm(axial + a_out.squeeze(1)),
            self.norm(coronal + c_out.squeeze(1)),
            self.norm(sagittal + s_out.squeeze(1)),
        ], dim=1)  # [B, 3*D]
        return fused


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class EfficientNetB0_ViT(nn.Module):
    """
    EfficientNetB0 + multi-layer SliceViT + Attention Pooling + Cross-Plane Fusion.

    Thay đổi so với v1:
      - SliceViT: num_layers > 1, DropPath, không raise error khi slices > max_slices.
      - Pool: AttentionPool thay vì max-pool.
      - Fusion: Cross-Attention giữa 3 plane thay vì concat thô.
      - Head: thêm BatchNorm + skip connection.
      - Hỗ trợ gradient checkpointing để giảm VRAM khi fine-tune backbone.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_vit_layers: int = 2,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        max_slices: int = 64,
        freeze_backbone: bool = False,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        feat_dim = 1280
        self.use_grad_checkpoint = use_grad_checkpoint and _HAS_GRAD_CKPT

        # --- Backbone (3 plane riêng biệt) ---
        self.axial = _build_efficientnet_b0_features()
        self.coronal = _build_efficientnet_b0_features()
        self.sagittal = _build_efficientnet_b0_features()
        self.pool_2d = nn.AdaptiveAvgPool2d(1)

        # --- Attention Pooling (thay max-pool) ---
        self.axial_apool = AttentionPool(feat_dim)
        self.coronal_apool = AttentionPool(feat_dim)
        self.sagittal_apool = AttentionPool(feat_dim)

        # --- SliceViT (multi-layer) ---
        vit_kwargs = dict(
            in_dim=feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_vit_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            max_slices=max_slices,
        )
        self.axial_vit = SliceViT(**vit_kwargs)
        self.coronal_vit = SliceViT(**vit_kwargs)
        self.sagittal_vit = SliceViT(**vit_kwargs)

        # Projection: đưa attention-pool feat_dim về embed_dim để cùng chiều với ViT
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
        )

        plane_dim = embed_dim * 2   # attn-pool + vit CLS
        self.fusion = PlanesCrossAttentionFusion = PlanesCF(plane_dim, num_heads=4, dropout=dropout)

        # --- Classification Head ---
        fused_dim = 3 * plane_dim
        self.head = _ClassHead(fused_dim, hidden=512, dropout=dropout)

        if freeze_backbone:
            self.freeze_backbones()

    # ------------------------------------------------------------------
    def freeze_backbones(self):
        for net in (self.axial, self.coronal, self.sagittal):
            for p in net.parameters():
                p.requires_grad = False

    def unfreeze_backbones(self):
        for net in (self.axial, self.coronal, self.sagittal):
            for p in net.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    def _extract_features(self, backbone, x: torch.Tensor) -> torch.Tensor:
        """Chạy backbone trên batch slices, trả về [B, S, feat_dim]."""
        if x.dim() == 4:
            # Single-sample path (S, C, H, W)
            if self.use_grad_checkpoint:
                feat = grad_checkpoint(backbone, x)
            else:
                feat = backbone(x)
            feat = self.pool_2d(feat).view(1, x.shape[0], -1)
            return feat

        if x.dim() != 5:
            raise ValueError(f"Expected 4-D or 5-D input, got {x.shape}")

        B, S, C, H, W = x.shape
        x_flat = x.reshape(B * S, C, H, W)
        if self.use_grad_checkpoint:
            feat = grad_checkpoint(backbone, x_flat)
        else:
            feat = backbone(x_flat)
        feat = self.pool_2d(feat).view(B, S, -1)
        return feat

    def _encode_plane(self, backbone, apool, vit, x: torch.Tensor) -> torch.Tensor:
        feat = self._extract_features(backbone, x)     # [B, S, 1280]
        pool_feat = self.pool_proj(apool(feat))        # [B, embed_dim]
        vit_feat = vit(feat)                           # [B, embed_dim]
        return torch.cat([pool_feat, vit_feat], dim=1) # [B, 2*embed_dim]

    # ------------------------------------------------------------------
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("Input must be [axial, coronal, sagittal].")

        a = self._encode_plane(self.axial,    self.axial_apool,    self.axial_vit,    x[0])
        c = self._encode_plane(self.coronal,  self.coronal_apool,  self.coronal_vit,  x[1])
        s = self._encode_plane(self.sagittal, self.sagittal_apool, self.sagittal_vit, x[2])

        fused = self.fusion(a, c, s)      # [B, 3 * 2*embed_dim]
        return self.head(fused)           # [B, 1]


# ---------------------------------------------------------------------------
# Fusion & Head helpers
# ---------------------------------------------------------------------------

class PlanesCF(nn.Module):
    """Cross-attention fusion cho 3 plane vectors."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ca = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, a, c, s):
        tokens = torch.stack([a, c, s], dim=1)  # [B, 3, D]
        out, _ = self.ca(tokens, tokens, tokens)
        out = self.norm(tokens + out)
        return out.reshape(out.size(0), -1)      # [B, 3*D]


class _ClassHead(nn.Module):
    """Head với LayerNorm (batch-size-agnostic), Residual và Dropout."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)   # LayerNorm: OK với batch=1
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.skip = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.ln1(self.fc1(x))))
        h = self.act(self.ln2(self.fc2(h))) + self.skip(x)  # residual
        return self.out(self.drop(h))