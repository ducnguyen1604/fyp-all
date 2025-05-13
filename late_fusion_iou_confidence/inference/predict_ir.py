# predict_ir.py

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.load_model import load_model
from utils.box_utils import xywh_to_xyxy
import json

def load_test_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    return sorted(files)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0), image_path

def decode_prediction(output, image_size, conf_thresh=0.3):
    _, _, h, w = output.shape
    output = output.squeeze(0).detach().cpu()
    pred_cls = output[0, :, :].mean().item()
    pred_box = output[1:, :, :].mean(dim=[1, 2]).tolist()

    if pred_cls < conf_thresh:
        return None

    x1, y1, x2, y2 = xywh_to_xyxy(pred_box)
    x1 = int(x1 * image_size[0])
    y1 = int(y1 * image_size[1])
    x2 = int(x2 * image_size[0])
    y2 = int(y2 * image_size[1])
    return {
        "bbox": [x1, y1, x2, y2],
        "score": pred_cls,
        "label": 0
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_type='ir', weights_path='results/logs/ir/epoch_50.pth', device=device)
    model.eval()

    test_folder = "data/images_ir"
    output_json = "results/fused_detections/preds_ir.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    results = []

    for img_name in load_test_images(test_folder):
        img_path = os.path.join(test_folder, img_name)
        input_tensor, _ = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            result = decode_prediction(output, image_size=(416, 416))

        if result:
            result['filename'] = img_name
            results.append(result)

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved IR predictions to {output_json}")

if __name__ == "__main__":
    main()
