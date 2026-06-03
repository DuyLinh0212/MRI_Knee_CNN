import torch
import torch.nn as nn
from torchvision import models


# =========================================================
# 1. Build EfficientNetB0 feature extractor
# =========================================================
def _build_efficientnet_b0_features():
    """
    Load EfficientNetB0 pretrained and return only feature extractor part.
    Output feature map has 1280 channels.
    """
    try:
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
    except Exception:
        net = models.efficientnet_b0(pretrained=True)

    return net.features


# =========================================================
# 2. Slice-level Transformer / ViT aggregator
# =========================================================
class SliceViT(nn.Module):
    """
    Lightweight Transformer Encoder over slice-level EfficientNet features.

    Input:
        x: [B, S, 1280]
        B = batch size
        S = number of slices
        1280 = EfficientNetB0 feature dimension

    Output:
        [B, embed_dim]
    """

    def __init__(
        self,
        in_dim: int = 1280,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        max_slices: int = 64,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [B, S, 1280]
        """

        if x.dim() != 3:
            raise ValueError(f"SliceViT expects input [B, S, C], got {x.shape}")

        batch_size, slices, _ = x.shape

        if slices + 1 > self.pos_embed.size(1):
            raise ValueError(
                f"Input has {slices} slices, "
                f"but max_slices is {self.pos_embed.size(1) - 1}."
            )

        # [B, S, 1280] -> [B, S, embed_dim]
        x = self.proj(x)

        # cls token: [B, 1, embed_dim]
        cls = self.cls_token.expand(batch_size, -1, -1)

        # [B, S + 1, embed_dim]
        x = torch.cat([cls, x], dim=1)

        # add positional embedding
        x = x + self.pos_embed[:, : slices + 1, :]

        # Transformer Encoder
        x = self.encoder(x)

        # return cls token representation
        return self.norm(x[:, 0])


# =========================================================
# 3. EfficientNetB0 + Shared Backbone + 3 SliceViT branches
# =========================================================
class EfficientNetB0_SharedBackbone_ViT(nn.Module):
    """
    EfficientNetB0 shared backbone + 3 lightweight SliceViT aggregators.

    Input:
        x = [axial, coronal, sagittal]

    Each plane shape:
        [B, S, C, H, W]

    Example:
        axial    = [B, S1, 3, 224, 224]
        coronal  = [B, S2, 3, 224, 224]
        sagittal = [B, S3, 3, 224, 224]

    Output:
        logits: [B, 1]

    Use with:
        nn.BCEWithLogitsLoss()
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        max_slices: int = 64,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        feat_dim = 1280

        # Chỉ dùng 1 EfficientNetB0 chung cho cả 3 mặt
        self.backbone = _build_efficientnet_b0_features()

        # Global Average Pooling: [N, 1280, H, W] -> [N, 1280, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Mỗi mặt vẫn có SliceViT riêng
        self.axial_vit = SliceViT(
            in_dim=feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_slices=max_slices,
        )

        self.coronal_vit = SliceViT(
            in_dim=feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_slices=max_slices,
        )

        self.sagittal_vit = SliceViT(
            in_dim=feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_slices=max_slices,
        )

        # Classifier sau khi concat 3 mặt
        self.fc = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        if freeze_backbone:
            self.freeze_backbone()

    # -----------------------------------------------------
    # Freeze / unfreeze backbone
    # -----------------------------------------------------
    def freeze_backbone(self):
        """
        Freeze EfficientNetB0 backbone.
        Nên dùng ở giai đoạn train đầu với dataset nhỏ.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze toàn bộ EfficientNetB0 backbone.
        Chỉ nên dùng khi fine-tune với learning rate nhỏ.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_backbone_blocks(self, num_blocks: int = 2):
        """
        Fine-tune nhẹ vài block cuối của EfficientNetB0.
        Phù hợp hơn so với mở toàn bộ backbone.

        Ví dụ:
            model.unfreeze_last_backbone_blocks(num_blocks=2)
        """

        # Freeze tất cả trước
        self.freeze_backbone()

        # Mở num_blocks block cuối
        blocks = list(self.backbone.children())

        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    # -----------------------------------------------------
    # Encode one MRI plane
    # -----------------------------------------------------
    def _encode_plane(self, vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Encode one plane: axial / coronal / sagittal.

        Accepted input:
            [B, S, C, H, W] for training
            [S, C, H, W] for single-case inference

        Output:
            [B, embed_dim]
        """

        # Single case: [S, C, H, W]
        if x.dim() == 4:
            slices = x.shape[0]

            feat = self.backbone(x)
            feat = self.pool(feat)

            # [S, 1280, 1, 1] -> [1, S, 1280]
            feat = feat.view(1, slices, -1)

            return vit(feat)

        # Batch case: [B, S, C, H, W]
        if x.dim() == 5:
            batch_size, slices, channels, height, width = x.shape

            # [B, S, C, H, W] -> [B*S, C, H, W]
            x = x.reshape(batch_size * slices, channels, height, width)

            feat = self.backbone(x)
            feat = self.pool(feat)

            # [B*S, 1280, 1, 1] -> [B, S, 1280]
            feat = feat.view(batch_size, slices, -1)

            return vit(feat)

        raise ValueError(
            f"Unexpected input shape for plane. "
            f"Expected [B,S,C,H,W] or [S,C,H,W], got {x.shape}"
        )

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, x):
        """
        x must be:
            [axial, coronal, sagittal]

        Each item:
            [B, S, C, H, W]
        """

        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(
                "Input must be a list/tuple: [axial, coronal, sagittal]."
            )

        axial_x, coronal_x, sagittal_x = x

        axial_feat = self._encode_plane(self.axial_vit, axial_x)
        coronal_feat = self._encode_plane(self.coronal_vit, coronal_x)
        sagittal_feat = self._encode_plane(self.sagittal_vit, sagittal_x)

        # [B, embed_dim] * 3 -> [B, 3 * embed_dim]
        feats = torch.cat(
            [axial_feat, coronal_feat, sagittal_feat],
            dim=1
        )

        # logits: [B, 1]
        logits = self.fc(feats)

        return logits


EfficientNetB0_ViT = EfficientNetB0_SharedBackbone_ViT


# =========================================================
# 4. Example usage
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0_SharedBackbone_ViT(
        embed_dim=128,
        num_heads=4,
        dropout=0.3,
        max_slices=64,
        freeze_backbone=True,
    ).to(device)

    # Example batch
    # B = 2 cases
    # S = 24 slices per plane
    # C = 3 channels
    # H = W = 224
    axial = torch.randn(2, 24, 3, 224, 224).to(device)
    coronal = torch.randn(2, 24, 3, 224, 224).to(device)
    sagittal = torch.randn(2, 24, 3, 224, 224).to(device)

    logits = model([axial, coronal, sagittal])

    print("Output logits shape:", logits.shape)  # [2, 1]

    # For binary classification
    probs = torch.sigmoid(logits)
    print("Output probs shape:", probs.shape)    # [2, 1]
