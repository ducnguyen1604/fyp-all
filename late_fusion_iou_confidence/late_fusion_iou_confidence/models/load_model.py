# load_model.py
import torch
from models.yolo_rgb import YOLOv11Model as YOLOv11RGB
from models.yolo_ir import YOLOv11Model as YOLOv11IR

def load_model(model_type='rgb', weights_path=None, num_classes=3, device='cuda'):
    if model_type == 'rgb':
        model = YOLOv11RGB(num_classes=num_classes)
    elif model_type == 'ir':
        model = YOLOv11IR(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    return model.to(device)
