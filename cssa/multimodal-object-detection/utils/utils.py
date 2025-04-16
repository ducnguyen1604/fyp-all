import torch

def train_epoch(backbone, cssa_module, dataloader, optimizer, device):
    backbone.train()
    cssa_module.train()

    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        features = backbone(images)
        outputs = cssa_module(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_checkpoint(backbone, cssa_module, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'backbone': backbone.state_dict(),
        'cssa': cssa_module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)
