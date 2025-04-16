# yolo_rgb.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class YOLOv11Head(nn.Module):
    def __init__(self, in_channels=2048, num_classes=3):
        super(YOLOv11Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes * 5, kernel_size=1)  # 5 = [x, y, w, h, conf]
        )

    def forward(self, x):
        return self.head(x)

class YOLOv11Model(nn.Module):
    def __init__(self, num_classes=3):
        super(YOLOv11Model, self).__init__()
        backbone = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # remove avgpool + fc
        self.head = YOLOv11Head(in_channels=2048, num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.head(features)
        return out
