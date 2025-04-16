# trainer_utils.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import json

class SimpleDataset(Dataset):
    def __init__(self, data_path, modality):
        self.image_dir = os.path.join(data_path, f"images_{modality}")
        self.ann_file = os.path.join(data_path, "annotations", f"{modality}.json")
        with open(self.ann_file, 'r') as f:
            self.annotations = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.image_dir, ann["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Placeholder: bbox should be [x1, y1, x2, y2] and class id
        bbox = torch.tensor(ann["bbox"], dtype=torch.float32)
        label = torch.tensor(ann["label"], dtype=torch.long)

        return image, bbox, label

def get_dataset(data_path, modality):
    return SimpleDataset(data_path, modality)

def train_model(model, dataloader, device, args):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_box = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for images, boxes, labels in dataloader:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Assume outputs shape: [B, num_classes * 5, H, W]
            # Placeholder: extract predicted class and bbox here
            pred_cls = outputs[:, :3, :, :].mean(dim=[2, 3])
            pred_bbox = outputs[:, 3:, :, :].mean(dim=[2, 3])

            loss_cls = criterion_cls(pred_cls, labels)
            loss_box = criterion_box(pred_bbox, boxes)
            loss = loss_cls + loss_box

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))
