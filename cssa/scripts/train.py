import torch
from torch.utils.data import DataLoader
from models.backbone import get_resnet50_backbone
from models.mobilenetv3 import get_mobilenetv3_backbone, get_feature_dim as mobilenet_feature_dim
from models.cssa import CSSA
from data.dataset import LLVIPDataset
from utils.utils import train_epoch, save_checkpoint
from configs.config import config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_backbone(name='resnet50', pretrained=True):
    if name.lower() == 'resnet50':
        backbone = get_resnet50_backbone(pretrained=pretrained)
        feature_dim = 2048
    elif name.lower() == 'mobilenetv3':
        backbone = get_mobilenetv3_backbone(pretrained=pretrained)
        feature_dim = mobilenet_feature_dim()
    else:
        raise ValueError(f"Unsupported backbone '{name}'")
    return backbone, feature_dim

def main(backbone_name='resnet50'):
    dataset = LLVIPDataset(root='data/raw', partition='train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    backbone, feature_dim = get_backbone(backbone_name)
    backbone = backbone.to(DEVICE)

    cssa_module = CSSA(in_channels=feature_dim,
                       reduction_factor=config['reduction_factor'],
                       cssa_thresh=config['cssa_switching_thresh']).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(cssa_module.parameters()), 
        lr=config['lr']
    )

    print(f"Training with backbone: {backbone_name}")

    for epoch in range(config['epochs']):
        loss = train_epoch(backbone, cssa_module, dataloader, optimizer, DEVICE)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {loss:.4f}")

        if (epoch + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint(backbone, cssa_module, optimizer, epoch, path=f"{backbone_name}_checkpoint_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train multimodal object detection model.')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'mobilenetv3'], help='Backbone model to use.')
    args = parser.parse_args()

    main(backbone_name=args.backbone)
