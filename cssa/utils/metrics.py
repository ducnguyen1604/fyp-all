import torch

def evaluate_model(backbone, cssa_module, dataloader, device):
    backbone.eval()
    cssa_module.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = cssa_module(features)
            preds = torch.argmax(outputs, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return {'accuracy': accuracy}
