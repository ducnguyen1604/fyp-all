import torchvision.models as models
import torch.nn as nn

def get_resnet50_backbone(pretrained=True, num_classes=2):
    model = models.resnet50(pretrained=pretrained)
    modules = list(model.children())[:-2]  # remove the final FC layers
    backbone = nn.Sequential(*modules)
    return backbone
