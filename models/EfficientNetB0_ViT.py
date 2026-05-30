import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0_features():
    try:
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        net = models.efficientnet_b0(pretrained=True)
    return net.features


class SliceViT(nn.Module):
    """A light one-layer Transformer over slice-level EfficientNet features."""

    def __init__(
        self,
        in_dim: int = 1280,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_slices: int = 64,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, 1280]
        batch_size, slices, _ = x.shape
        if slices + 1 > self.pos_embed.size(1):
            raise ValueError(
                f"Input has {slices} slices, but max_slices is {self.pos_embed.size(1) - 1}."
            )

        x = self.proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : slices + 1, :]
        x = self.encoder(x)
        return self.norm(x[:, 0])


class EfficientNetB0_ViT(nn.Module):
    """EfficientNetB0 baseline with a small one-layer ViT slice aggregator."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        dropout: float = 0.15,
        max_slices: int = 64,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        feat_dim = 1280

        self.axial = _build_efficientnet_b0_features()
        self.coronal = _build_efficientnet_b0_features()
        self.sagittal = _build_efficientnet_b0_features()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.axial_vit = SliceViT(feat_dim, embed_dim, num_heads, dropout, max_slices)
        self.coronal_vit = SliceViT(feat_dim, embed_dim, num_heads, dropout, max_slices)
        self.sagittal_vit = SliceViT(feat_dim, embed_dim, num_heads, dropout, max_slices)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * embed_dim, 1),
        )

        if freeze_backbone:
            self.freeze_backbones()

    def freeze_backbones(self):
        for net in (self.axial, self.coronal, self.sagittal):
            for param in net.parameters():
                param.requires_grad = False

    def unfreeze_backbones(self):
        for net in (self.axial, self.coronal, self.sagittal):
            for param in net.parameters():
                param.requires_grad = True

    def _encode_plane(self, backbone, vit, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            slices = x.shape[0]
            feat = backbone(x)
            feat = self.pool(feat).view(1, slices, -1)
            return vit(feat)

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        batch_size, slices, channels, height, width = x.shape
        x = x.reshape(batch_size * slices, channels, height, width)
        feat = backbone(x)
        feat = self.pool(feat).view(batch_size, slices, -1)
        return vit(feat)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("Input must be a list/tuple: [axial, coronal, sagittal].")

        axial = self._encode_plane(self.axial, self.axial_vit, x[0])
        coronal = self._encode_plane(self.coronal, self.coronal_vit, x[1])
        sagittal = self._encode_plane(self.sagittal, self.sagittal_vit, x[2])
        feats = torch.cat([axial, coronal, sagittal], dim=1)
        return self.fc(feats)