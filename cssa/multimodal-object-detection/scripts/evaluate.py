import torch
from torch.utils.data import DataLoader
from models.backbone import get_resnet50_backbone
from models.cssa import CSSA
from data.dataset import LLVIPDataset
from utils.metrics import evaluate_model
from configs.config import config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    dataset = LLVIPDataset(root='data/raw', partition='val')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    backbone = get_resnet50_backbone(pretrained=False).to(DEVICE)
    cssa_module = CSSA(in_channels=2048, 
                       reduction_factor=config['reduction_factor'],
                       cssa_thresh=config['cssa_switching_thresh']).to(DEVICE)

    checkpoint = torch.load(config['checkpoint_path'])
    backbone.load_state_dict(checkpoint['backbone'])
    cssa_module.load_state_dict(checkpoint['cssa'])

    metrics = evaluate_model(backbone, cssa_module, dataloader, DEVICE)
    print(f"Evaluation metrics: {metrics}")

if __name__ == '__main__':
    main()
