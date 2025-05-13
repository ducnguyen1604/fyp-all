# train_ir.py

import argparse
import torch
from torch.utils.data import DataLoader
from models.yolo_ir import YOLOv11Model
from train.trainer_utils import get_dataset, train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='results/logs/ir/')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv11Model().to(device)

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    train_loader = DataLoader(
        get_dataset(args.data_path, modality='ir'),
        batch_size=args.batch_size,
        shuffle=True
    )

    train_model(model, train_loader, device, args)

if __name__ == "__main__":
    main()
