import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    return models.efficientnet_b0(weights=None).features


def _build_densenet121():
    return models.densenet121(weights=None).features


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 backbone for 3-plane knee MRI volumes."""

    def __init__(self):
        super().__init__()
        self.axial = _build_efficientnet_b0()
        self.coronal = _build_efficientnet_b0()
        self.sagittal = _build_efficientnet_b0()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3 * 1280, 1)

    def _encode_plane(self, net, x):
        if x.dim() == 4:
            feat = net(x)
            feat = self.pool(feat).view(feat.size(0), -1)
            return torch.max(feat, dim=0, keepdim=True)[0]

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        batch_size, slices, channels, height, width = x.shape
        x = x.view(batch_size * slices, channels, height, width)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = feat.view(batch_size, slices, -1)
        return torch.max(feat, dim=1)[0]

    def forward(self, images):
        axial = self._encode_plane(self.axial, images[0])
        coronal = self._encode_plane(self.coronal, images[1])
        sagittal = self._encode_plane(self.sagittal, images[2])
        feats = torch.cat([axial, coronal, sagittal], dim=1)
        return self.fc(feats)


class Densenet121(nn.Module):
    """DenseNet121 backbone for 3-plane knee MRI volumes."""

    def __init__(self):
        super().__init__()
        self.axial = _build_densenet121()
        self.coronal = _build_densenet121()
        self.sagittal = _build_densenet121()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3 * 1024, 1)

    def _encode_plane(self, net, x):
        if x.dim() == 4:
            feat = net(x)
            feat = self.pool(feat).view(feat.size(0), -1)
            return torch.max(feat, dim=0, keepdim=True)[0]

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        batch_size, slices, channels, height, width = x.shape
        x = x.view(batch_size * slices, channels, height, width)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = feat.view(batch_size, slices, -1)
        return torch.max(feat, dim=1)[0]

    def forward(self, images):
        axial = self._encode_plane(self.axial, images[0])
        coronal = self._encode_plane(self.coronal, images[1])
        sagittal = self._encode_plane(self.sagittal, images[2])
        feats = torch.cat([axial, coronal, sagittal], dim=1)
        return self.fc(feats)


class EfficientNetB0ViT(nn.Module):
    """EfficientNet-B0 extracts slice features, Transformer encodes slice sequence."""

    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, max_slices=64):
        super().__init__()
        self.axial = _build_efficientnet_b0()
        self.coronal = _build_efficientnet_b0()
        self.sagittal = _build_efficientnet_b0()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(3 * embed_dim, 1)

    def _encode_plane(self, net, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        batch_size, slices, channels, height, width = x.shape
        x = x.view(batch_size * slices, channels, height, width)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = self.proj(feat).view(batch_size, slices, -1)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.transformer(tokens)
        return tokens[:, 0]

    def forward(self, images):
        axial = self._encode_plane(self.axial, images[0])
        coronal = self._encode_plane(self.coronal, images[1])
        sagittal = self._encode_plane(self.sagittal, images[2])
        feats = torch.cat([axial, coronal, sagittal], dim=1)
        return self.fc(feats)
