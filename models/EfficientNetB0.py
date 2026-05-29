import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    """Tạo backbone EfficientNet-B0 với trọng số pretrained khi có thể."""
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


class EfficientNetB0(nn.Module):
    """Backbone EfficientNet-B0 xử lý MRI theo 3 mặt cắt."""

    def __init__(self):
        super().__init__()

        self.axial = _build_efficientnet_b0().features
        self.coronal = _build_efficientnet_b0().features
        self.sagittal = _build_efficientnet_b0().features

        # Kích thước đặc trưng cuối của EfficientNet-B0 là 1280.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3 * 1280, 1)

    def _encode_plane(self, net, x):
        """Mã hóa một mặt cắt MRI và gộp đặc trưng theo chiều slice."""
        # x có thể là [S, C, H, W] hoặc [B, S, C, H, W].
        if x.dim() == 4:
            feat = net(x)
            feat = self.pool(feat).view(feat.size(0), -1)
            feat = torch.max(feat, dim=0, keepdim=True)[0]
            return feat

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = net(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = feat.view(b, s, -1)
        feat = torch.max(feat, dim=1)[0]
        return feat

    def forward(self, x):
        # Nhận list 3 tensor tương ứng axial, coronal, sagittal.
        images = x
        axial = self._encode_plane(self.axial, images[0])
        coronal = self._encode_plane(self.coronal, images[1])
        sagittal = self._encode_plane(self.sagittal, images[2])

        feats = torch.cat([axial, coronal, sagittal], dim=1)
        output = self.fc(feats)
        return output
