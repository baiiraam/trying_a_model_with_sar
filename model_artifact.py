import torch
import torch.nn as nn
import torchvision.models as models


class SSL4EOS12DeCURModel(nn.Module):
    """ResNet-50 model pretrained with DeCUR on SSL4EO-S12 dataset"""

    def __init__(self, in_channels=2, num_classes=1):
        super().__init__()

        # Load ResNet-50 without pretrained weights
        self.backbone = models.resnet50(weights=None)

        # Modify first conv layer for SAR input (2 bands: VV, VH)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove original fc layer, keep features up to avgpool
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Add segmentation head for water detection
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.seg_head(features)
        return torch.sigmoid(output)
