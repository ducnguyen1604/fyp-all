# main.py

import os
import torch
from PIL import Image
from torchvision import transforms
from models.load_model import load_model
from fusion.fuse_detections import fuse_detections
from utils.visualizer import draw_boxes, save_image

# --- CONFIGURATION ---
RGB_MODEL_PATH = 'results/logs/rgb/best_model.pth'
IR_MODEL_PATH = 'results/logs/ir/best_model.pth'
RGB_IMAGE_DIR = 'data/images_rgb'
IR_IMAGE_DIR = 'data/images_ir'
OUTPUT_DIR = 'results/fused_detections'
ALPHA = 0.6
IOU_THRESH = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Models ---
model_rgb = load_model('rgb', RGB_MODEL_PATH, num_classes=1, device=DEVICE)
model_ir = load_model('ir', IR_MODEL_PATH, num_classes=1, device=DEVICE)
model_rgb.eval()
model_ir.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

def predict(model, image):
    with torch.no_grad():
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        output = model(image_tensor)[0]  # shape: [C, H, W]
        return parse_output(output)

def parse_output(output_tensor, score_thresh=0.3):
    """
    Converts model output into list of box dicts.
    Each box dict: {bbox, score, label}
    """
    output_tensor = output_tensor.cpu()
    boxes = []
    C, H, W = output_tensor.shape

    cls_scores = torch.sigmoid(output_tensor[0])  # 1 class
    bbox_map = output_tensor[1:5]  # x, y, w, h

    for i in range(H):
        for j in range(W):
            score = cls_scores[i, j].item()
            if score > score_thresh:
                x_center = (j + 0.5) * (416 / W)
                y_center = (i + 0.5) * (416 / H)
                w = bbox_map[2, i, j].item() * 416
                h = bbox_map[3, i, j].item() * 416

                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2

                boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': score,
                    'label': 0
                })
    return boxes

# --- Main Execution Loop ---
image_names = sorted(os.listdir(RGB_IMAGE_DIR))
for name in image_names:
    if not name.lower().endswith(('.jpg', '.png')):
        continue

    rgb_path = os.path.join(RGB_IMAGE_DIR, name)
    ir_path = os.path.join(IR_IMAGE_DIR, name)

    if not os.path.exists(ir_path):
        print(f"[WARN] Missing IR image for {name}, skipping.")
        continue

    rgb_img = Image.open(rgb_path).convert("RGB")
    ir_img = Image.open(ir_path).convert("RGB")

    rgb_preds = predict(model_rgb, rgb_img)
    ir_preds = predict(model_ir, ir_img)

    fused_preds = fuse_detections(rgb_preds, ir_preds, alpha=ALPHA, iou_thresh=IOU_THRESH)

    drawn = draw_boxes(rgb_img.copy(), fused_preds, label_map={0: "object"}, box_color="fused")
    save_image(os.path.join(OUTPUT_DIR, name), drawn)
    print(f"[✓] Processed {name} → {len(fused_preds)} fused detections")

print("✅ Fusion complete. Results saved to:", OUTPUT_DIR)
